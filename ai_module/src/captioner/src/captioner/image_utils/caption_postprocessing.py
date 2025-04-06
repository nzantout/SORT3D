PREFIX_STRINGS = [
    "view of "
]

def filter_prefix(caption: str):
    for prefix in PREFIX_STRINGS:
        prefix_idx = caption.find(prefix)
        if prefix_idx > -1:
            return caption[prefix_idx + len(prefix):]


def postprocess_captions(captions: list[str]):

    captions_out = [filter_prefix(caption) for caption in captions]

    return captions_out
