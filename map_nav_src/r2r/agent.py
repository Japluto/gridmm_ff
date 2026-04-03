import json
import os
import sys
import numpy as np
import random
import math
import time
import re
from collections import defaultdict
try:
    import line_profiler  # noqa: F401
except ImportError:
    line_profiler = None

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from .agent_base import Seq2SeqAgent
from .eval_utils import cal_dtw

from models.graph_utils import GraphMap
from models.model import VLNBert, Critic
from models.ops import pad_tensors_wgrad


class GMapNavAgent(Seq2SeqAgent):
    
    def _build_model(self):
        self.vln_bert = VLNBert(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        # buffer
        self.scanvp_cands = {}
        self.instr_rerank_stats = defaultdict(int)
        self.anti_loop_stats = defaultdict(int)
        self._instr_rerank_debug_examples_printed = 0

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None, viz=False):
        self.instr_rerank_stats = defaultdict(int)
        self.anti_loop_stats = defaultdict(int)
        self._instr_rerank_debug_examples_printed = 0
        super().test(use_dropout=use_dropout, feedback=feedback, allow_cheat=allow_cheat, iters=iters, viz=viz)
        if self.args.instr_rerank_enabled and self.args.test and self.default_gpu:
            print(
                '[instr_rerank_stats] '
                f"rerank_trigger_count={self.instr_rerank_stats['rerank_trigger_count']}, "
                f"rerank_action_changed_count={self.instr_rerank_stats['rerank_action_changed_count']}, "
                f"left_cue_count={self.instr_rerank_stats['left_cue_count']}, "
                f"right_cue_count={self.instr_rerank_stats['right_cue_count']}, "
                f"straight_cue_count={self.instr_rerank_stats['straight_cue_count']}, "
                f"first_cue_used_count={self.instr_rerank_stats['first_cue_used_count']}, "
                f"local_clause_used_count={self.instr_rerank_stats['local_clause_used_count']}, "
                f"conflict_disabled_count={self.instr_rerank_stats['conflict_disabled_count']}, "
                f"no_cue_disabled_count={self.instr_rerank_stats['no_cue_disabled_count']}, "
                f"topk_match_count={self.instr_rerank_stats['topk_match_count']}, "
                f"topk_rerank_applied_count={self.instr_rerank_stats['topk_rerank_applied_count']}"
            )
        if self.args.anti_loop_enabled and self.args.test and self.default_gpu:
            print(
                '[anti_loop_stats] '
                f"anti_loop_trigger_count={self.anti_loop_stats['anti_loop_trigger_count']}, "
                f"backtrack_penalty_count={self.anti_loop_stats['backtrack_penalty_count']}, "
                f"revisit_penalty_count={self.anti_loop_stats['revisit_penalty_count']}, "
                f"anti_loop_action_changed_count={self.anti_loop_stats['anti_loop_action_changed_count']}"
            )

    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]
        
        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=bool)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask
        }

    def _panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []
        
        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)
            
            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts, 
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens, 
            'cand_vpids': batch_cand_vpids,
        }

    def _nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)
        
        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []                
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes:
                    if k == obs[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )   # cuda

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i+1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds, 
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks, 
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
            'grid_fts': [obs[index]['grid_fts'].cuda() for index in range(len(obs))],
            'grid_map': [obs[index]['grid_map'].cuda() for index in range(len(obs))],
            'gridmap_pos_fts': torch.cat([obs[index]['gridmap_pos_fts'].unsqueeze(0).cuda() for index in range(len(obs))], dim=0),
            'grid_age': [obs[index]['grid_age'].cuda() for index in range(len(obs))],
            'grid_visit_count': [obs[index]['grid_visit_count'].cuda() for index in range(len(obs))],
            'grid_novelty_ema': [obs[index]['grid_novelty_ema'].cuda() for index in range(len(obs))],
        }

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i], 
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp], 
                obs[i]['heading'], obs[i]['elevation']
            )                    
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens+1),
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None]+x for x in cand_vpids],
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0    # Stop if arrived 
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                    + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                path_segment = gmaps[i].graph.path(ob['viewpoint'], action)
                for vp in path_segment:
                    traj[i]['path'].append([vp])
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    def _get_instruction_cue_source(self, instruction):
        if self.args.instr_rerank_cue_source_mode != 'local_first_clause':
            return instruction, False

        lowered = instruction.lower()
        boundary_patterns = [r'\.', r',', r';', r'\s+and then\s+', r'\s+then\s+']
        first_boundary = None
        for pattern in boundary_patterns:
            match = re.search(pattern, lowered)
            if match is None:
                continue
            if first_boundary is None or match.start() < first_boundary:
                first_boundary = match.start()

        if first_boundary is None:
            return instruction, False
        return instruction[:first_boundary].strip(), True

    def _extract_direction_cue(self, instruction):
        cue_source, used_local_clause = self._get_instruction_cue_source(instruction)
        lowered = cue_source.lower()
        phrase_patterns = (
            ('left', r'(?<!\w)turn\s+left(?!\w)'),
            ('right', r'(?<!\w)turn\s+right(?!\w)'),
            ('straight', r'(?<!\w)(?:go|walk|head)\s+straight(?!\w)'),
        )
        matches = []
        for cue, pattern in phrase_patterns:
            for match in re.finditer(pattern, lowered):
                matches.append((match.start(), cue))
        if len(matches) == 0:
            return None, 'no_cue', cue_source, used_local_clause
        matches.sort(key=lambda x: x[0])
        return matches[0][1], None, cue_source, used_local_clause

    def _bucket_heading_direction(self, heading):
        if abs(float(heading)) <= self.args.instr_rerank_heading_straight_thresh:
            return 'straight'
        if heading < -self.args.instr_rerank_heading_straight_thresh:
            return 'left'
        return 'right'

    def _get_graph_path_hops(self, gmap, cur_vp, target_vp):
        path = gmap.graph.path(cur_vp, target_vp)
        next_hop = path[0] if len(path) > 0 else None
        prev_hop = None
        if len(path) == 1:
            prev_hop = cur_vp
        elif len(path) > 1:
            prev_hop = path[-2]
        return path, next_hop, prev_hop

    def _apply_instruction_rerank(self, obs, nav_probs, nav_logits, nav_vpids, t, ended):
        if (not self.args.instr_rerank_enabled) or (not self.args.test):
            return nav_probs
        if self.feedback != 'argmax' or t >= self.args.instr_rerank_max_step:
            return nav_probs

        adjusted_nav_probs = nav_probs.clone()
        cpu_nav_probs = nav_probs.detach().cpu()
        cpu_nav_logits = nav_logits.detach().cpu()

        for i, ob in enumerate(obs):
            if ended[i]:
                continue

            original_top_idx = int(torch.argmax(cpu_nav_probs[i]).item())
            if original_top_idx == 0:
                continue

            cue, disable_reason, cue_source, used_local_clause = self._extract_direction_cue(ob['instruction'])
            if disable_reason == 'no_cue':
                self.instr_rerank_stats['no_cue_disabled_count'] += 1
                continue
            if disable_reason == 'conflict':
                self.instr_rerank_stats['conflict_disabled_count'] += 1
                continue
            self.instr_rerank_stats[f'{cue}_cue_count'] += 1
            self.instr_rerank_stats['first_cue_used_count'] += 1
            if used_local_clause:
                self.instr_rerank_stats['local_clause_used_count'] += 1

            valid_nonstop_indices = [
                j for j in range(1, len(nav_vpids[i]))
                if torch.isfinite(cpu_nav_logits[i, j]).item()
            ]
            if len(valid_nonstop_indices) < 2:
                continue

            valid_nonstop_probs = sorted(
                [float(cpu_nav_probs[i, j].item()) for j in valid_nonstop_indices],
                reverse=True
            )
            if valid_nonstop_probs[0] - valid_nonstop_probs[1] > self.args.instr_rerank_uncertainty_thresh:
                continue

            topk_nonstop_indices = set(
                sorted(
                    valid_nonstop_indices,
                    key=lambda j: float(cpu_nav_probs[i, j].item()),
                    reverse=True
                )[:self.args.instr_rerank_topk]
            )

            vp_to_score_idx = {vp: j for j, vp in enumerate(nav_vpids[i]) if vp is not None}
            matching_indices = []
            for cand in ob['candidate']:
                score_idx = vp_to_score_idx.get(cand['viewpointId'])
                if score_idx is None:
                    continue
                if not torch.isfinite(cpu_nav_logits[i, score_idx]).item():
                    continue
                if score_idx not in topk_nonstop_indices:
                    continue
                if self._bucket_heading_direction(cand['heading']) == cue:
                    matching_indices.append(score_idx)

            if not matching_indices:
                continue

            matching_indices = sorted(set(matching_indices))
            self.instr_rerank_stats['topk_match_count'] += 1
            adjusted_nav_probs[i, matching_indices] += self.args.instr_rerank_dir_boost
            self.instr_rerank_stats['rerank_trigger_count'] += 1
            self.instr_rerank_stats['topk_rerank_applied_count'] += 1

            reranked_top_idx = int(torch.argmax(adjusted_nav_probs[i]).item())
            if reranked_top_idx != original_top_idx:
                self.instr_rerank_stats['rerank_action_changed_count'] += 1

            if self.args.instr_rerank_debug_print and self._instr_rerank_debug_examples_printed < 3:
                print('[instr_rerank] instruction:', ob['instruction'])
                print('[instr_rerank] cue_source:', cue_source)
                print('[instr_rerank] cue:', cue)
                print('[instr_rerank] activated: True')
                print('[instr_rerank] original_top_idx:', original_top_idx)
                print('[instr_rerank] reranked_top_idx:', reranked_top_idx)
                self._instr_rerank_debug_examples_printed += 1

        return adjusted_nav_probs

    def _apply_anti_loop_penalty(self, nav_probs, nav_logits, nav_vpids, gmaps, obs, t, ended, prev_vpids, vp_visit_counts):
        if (not self.args.anti_loop_enabled) or (not self.args.test):
            return nav_probs
        if self.feedback != 'argmax' or t < self.args.anti_loop_min_step:
            return nav_probs

        adjusted_nav_probs = nav_probs.clone()
        cpu_nav_probs = nav_probs.detach().cpu()
        cpu_nav_logits = nav_logits.detach().cpu()

        for i, ob in enumerate(obs):
            if ended[i]:
                continue

            original_top_idx = int(torch.argmax(cpu_nav_probs[i]).item())
            step_triggered = False

            for j in range(1, len(nav_vpids[i])):
                if not torch.isfinite(cpu_nav_logits[i, j]).item():
                    continue
                target_vpid = nav_vpids[i][j]
                if target_vpid is None:
                    continue

                _, next_hop, _ = self._get_graph_path_hops(gmaps[i], ob['viewpoint'], target_vpid)
                if next_hop is None:
                    continue

                penalty = 0.0
                if prev_vpids[i] is not None and next_hop == prev_vpids[i]:
                    penalty += self.args.anti_loop_backtrack_penalty
                    self.anti_loop_stats['backtrack_penalty_count'] += 1
                if vp_visit_counts[i][next_hop] >= self.args.anti_loop_revisit_thresh:
                    penalty += self.args.anti_loop_revisit_penalty
                    self.anti_loop_stats['revisit_penalty_count'] += 1

                if penalty <= 0:
                    continue

                adjusted_nav_probs[i, j] -= penalty
                step_triggered = True

            if step_triggered:
                self.anti_loop_stats['anti_loop_trigger_count'] += 1
                reranked_top_idx = int(torch.argmax(adjusted_nav_probs[i]).item())
                if reranked_top_idx != original_top_idx:
                    self.anti_loop_stats['anti_loop_action_changed_count'] += 1

        return adjusted_nav_probs

    def _apply_dual_stop(self, a_t, nav_probs, nav_logits, nav_inputs, obs, t, ended, vp_visit_counts):
        dual_stop_vetoed = np.array([False] * len(obs))
        if (not self.args.dual_stop_enabled) or (not self.args.test):
            return a_t, dual_stop_vetoed
        if self.feedback in ('teacher', 'sample'):
            return a_t, dual_stop_vetoed

        cpu_nav_probs = nav_probs.detach().cpu()
        cpu_nav_logits = nav_logits.detach().cpu()

        for i, ob in enumerate(obs):
            if ended[i] or int(a_t[i].item()) != 0:
                continue

            stop_prob = float(cpu_nav_probs[i, 0].item())
            semantic_stop_ok = stop_prob >= self.args.dual_stop_score_thresh

            finite_mask = torch.isfinite(cpu_nav_logits[i])
            best_nonstop_idx = None
            best_nonstop_prob = -float('inf')
            for j in range(1, cpu_nav_probs.size(1)):
                if finite_mask[j]:
                    cand_prob = float(cpu_nav_probs[i, j].item())
                    if cand_prob > best_nonstop_prob:
                        best_nonstop_prob = cand_prob
                        best_nonstop_idx = j

            geometric_stop_ok = t >= self.args.dual_stop_min_step
            if geometric_stop_ok:
                geometric_stop_ok = (
                    vp_visit_counts[i][ob['viewpoint']] >= self.args.dual_stop_revisit_thresh or
                    nav_inputs['no_vp_left'][i] or
                    (
                        best_nonstop_idx is not None and
                        (stop_prob - best_nonstop_prob) >= self.args.dual_stop_margin_thresh
                    )
                )

            if semantic_stop_ok and geometric_stop_ok:
                continue

            if best_nonstop_idx is not None:
                a_t[i] = best_nonstop_idx
                dual_stop_vetoed[i] = True

        return a_t, dual_stop_vetoed

    # @profile
    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)
        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
        } for ob in obs]

        # Language input: txt_ids, txt_masks
        language_inputs = self._language_variable(obs)
        txt_embeds = self.vln_bert('language', language_inputs)
    
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)
        vp_visit_counts = [defaultdict(int) for _ in range(batch_size)]
        prev_vpids = [None] * batch_size

        # Init the logs
        masks = []
        entropys = []
        ml_loss = 0.     

        for t in range(self.args.max_action_len):
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    vp_visit_counts[i][obs[i]['viewpoint']] += 1
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                self._nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'], 
                    pano_inputs['view_lens'], pano_inputs['nav_types'],
                )
            )
            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
            })
            nav_outs = self.vln_bert('navigation', nav_inputs)

            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            grid_logits = nav_outs['grid_logits']
            nav_probs = torch.softmax(nav_logits, 1)
                                        
            if train_ml is not None:
                # Supervised training
                nav_targets = self._teacher_action(
                    obs, nav_vpids, ended, 
                    visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None
                )
                # print(t, nav_logits, nav_targets)
                #stop_logits = nav_logits[nav_targets==0]
                #stop_targets = nav_targets[nav_targets==0]

                #if stop_logits.shape[0] != 0:                    
                #    ml_loss += self.criterion(nav_logits, nav_targets) + self.criterion(stop_logits,stop_targets) * 2.
                #else:
                ml_loss += self.criterion(nav_logits, nav_targets)
                # print(t, 'ml_loss', ml_loss.item(), self.criterion(nav_logits, nav_targets).item())
                                                 
            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets                 # teacher forcing
 
            elif self.feedback == 'argmax':
                reranked_nav_probs = self._apply_instruction_rerank(
                    obs, nav_probs, nav_logits, nav_vpids, t, ended
                )
                reranked_nav_probs = self._apply_anti_loop_penalty(
                    reranked_nav_probs, nav_logits, nav_vpids, gmaps, obs, t, ended, prev_vpids, vp_visit_counts
                )
                _, a_t = reranked_nav_probs.max(1)        # student forcing - argmax
                a_t = a_t.detach() 
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach() 
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            a_t, dual_stop_vetoed = self._apply_dual_stop(
                a_t, nav_probs, nav_logits, nav_inputs, obs, t, ended, vp_visit_counts
            )

            # Keep historical stop scores consistent with the dual-stop veto.
            for i, gmap in enumerate(gmaps):
                if ended[i]:
                    continue
                if dual_stop_vetoed[i]:
                    continue
                i_vp = obs[i]['viewpoint']
                gmap.node_stop_scores[i_vp] = {
                    'stop': nav_probs[i, 0].data.item(),
                }

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample': # in training
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []  
            next_prev_vpids = list(prev_vpids)
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    target_vpid = nav_vpids[i][a_t[i]]
                    cpu_a_t.append(target_vpid)
                    _, _, prev_hop = self._get_graph_path_hops(gmaps[i], obs[i]['viewpoint'], target_vpid)
                    next_prev_vpids[i] = prev_hop

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        path_segment = gmaps[i].graph.path(obs[i]['viewpoint'], stop_node)
                        for vp in path_segment:
                            traj[i]['path'].append([vp])
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }

            # new observation and update graph
            obs = self.env._get_obs()
            prev_vpids = next_prev_vpids
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break

        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            self.loss += ml_loss
            self.logs['IL_loss'].append(ml_loss.item())

        return traj
