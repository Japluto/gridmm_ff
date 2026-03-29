import os
import json
import numpy as np
import re


INSTR_AUG_KEYWORDS = [
    'living room', 'dining room', 'bathroom', 'bedroom', 'hallway', 'kitchen',
    'stairway', 'stairs', 'doorway', 'fireplace', 'cabinet', 'picture',
    'window', 'closet', 'counter', 'shelf', 'table', 'chair', 'couch',
    'sofa', 'sink', 'lamp', 'desk', 'door', 'bed', 'left', 'right',
    'straight', 'forward', 'around', 'enter', 'exit', 'through', 'past',
    'across', 'into', 'down', 'stop', 'out', 'up'
]


def _encode_instruction(tokenizer_obj, instruction, max_instr_len):
    return tokenizer_obj.encode(
        instruction,
        add_special_tokens=True,
        truncation=True,
        max_length=max_instr_len,
    )


def _extract_keyword_summary(instruction, max_keywords):
    lowered = instruction.lower()
    matches = []
    for keyword in INSTR_AUG_KEYWORDS:
        pattern = r'(?<!\w)%s(?!\w)' % re.escape(keyword)
        for match in re.finditer(pattern, lowered):
            matches.append((match.start(), -len(keyword), keyword))

    if not matches:
        return []

    matches.sort()
    seen = set()
    ordered_keywords = []
    for _, _, keyword in matches:
        if keyword in seen:
            continue
        seen.add(keyword)
        ordered_keywords.append(keyword)
        if len(ordered_keywords) >= max_keywords:
            break
    return ordered_keywords


def _augment_instruction(instruction, max_keywords):
    keywords = _extract_keyword_summary(instruction, max_keywords)
    if len(keywords) < 2:
        return instruction, keywords
    return f"{instruction} [SEP] key cues: {' '.join(keywords)}", keywords


def _maybe_check_encoding_compatibility(data, tokenizer_obj, max_instr_len):
    # In debug mode, verify the live tokenizer reproduces the stored ids before augmentation.
    check_count = 0
    for item in data:
        for j, instr in enumerate(item['instructions']):
            stored = item['instr_encodings'][j][:max_instr_len]
            reencoded = _encode_instruction(tokenizer_obj, instr, max_instr_len)
            if stored != reencoded:
                raise RuntimeError(
                    'Instruction augmentation sanity check failed: stored instr_encoding '
                    'does not match tokenizer re-encoding for R2R.'
                )
            check_count += 1
            if check_count >= 5:
                return

def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if tokenizer == 'bert':
                filepath = os.path.join(anno_dir, '%s_%s_enc.json' % (dataset.upper(), split))
            elif tokenizer == 'xlm':
                filepath = os.path.join(anno_dir, '%s_%s_enc_xlmr.json' % (dataset.upper(), split))
            else:
                raise NotImplementedError('unspported tokenizer %s' % tokenizer)

            with open(filepath) as f:
                new_data = json.load(f)

            if split == 'val_train_seen':
                new_data = new_data[:50]

        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)
        # Join
        data += new_data
    return data

def construct_instrs(
    anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True,
    tokenizer_obj=None, args=None
):
    data = []
    loaded_data = load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test)

    instr_aug_enabled = bool(getattr(args, 'instr_aug_enabled', False))
    instr_aug_debug = bool(getattr(args, 'instr_aug_debug_print', False))
    instr_aug_max_keywords = int(getattr(args, 'instr_aug_max_keywords', 6))
    use_instr_aug = dataset == 'r2r' and is_test and instr_aug_enabled

    if use_instr_aug and tokenizer_obj is None:
        raise RuntimeError('Instruction augmentation requires a tokenizer object.')
    if use_instr_aug and instr_aug_debug:
        _maybe_check_encoding_compatibility(loaded_data, tokenizer_obj, max_instr_len)

    debug_examples_printed = 0
    for i, item in enumerate(loaded_data):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
            if use_instr_aug:
                aug_instr, keywords = _augment_instruction(instr, instr_aug_max_keywords)
                if aug_instr != instr:
                    new_item['instruction'] = aug_instr
                    new_item['instr_encoding'] = _encode_instruction(tokenizer_obj, aug_instr, max_instr_len)
                    if instr_aug_debug and debug_examples_printed < 3:
                        print('[instr_aug] original:', instr)
                        print('[instr_aug] augmented:', aug_instr)
                        debug_examples_printed += 1
            del new_item['instructions']
            del new_item['instr_encodings']
            data.append(new_item)
    return data
