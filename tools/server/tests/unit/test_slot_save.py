import pytest
from utils import *
import base64
import requests
import struct

# sequence state file: magic(4) version(4) payload_size(4), then payload_size llama_token words
STATE_FILE_HEADER_SIZE = 12

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = str(tmp_path)
    server.temperature = 0.0


def test_slot_save_restore():
    global server
    server.start()

    # First prompt in slot 1 should be fully processed
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # Save state of slot 1
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    # Since we have cache, this should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # Loading the saved cache into slot 0
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84

    # Since we have cache, slot 0 should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # For verification that slot 1 was not corrupted during slot 0 load, same thing should work
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_restore_legacy_token_list():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot_legacy.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    # rewrite the token payload into a plain token list, as written by servers that predate the packed server_tokens format
    path = os.path.join(server.slot_save_path, "slot_legacy.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())

    # the payload written by this server starts with a packed header: LLAMA_TOKEN_NULL(4) version(4) n_tokens(4)
    packed_header_size = 12

    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    payload_end = STATE_FILE_HEADER_SIZE + payload_size * 4
    n_tokens = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE + 8)[0]
    assert n_tokens == 84

    tokens_start = STATE_FILE_HEADER_SIZE + packed_header_size
    data = data[:STATE_FILE_HEADER_SIZE] + data[tokens_start:tokens_start + n_tokens * 4] + data[payload_end:]
    struct.pack_into("=I", data, STATE_FILE_HEADER_SIZE - 4, n_tokens)

    with open(path, "wb") as f:
        f.write(data)

    # the plain token list must restore, and the restored KV must be reusable
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot_legacy.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 6  # only the different part is processed



def test_slot_erase():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # erase slot 1
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    # re-run the same prompt, it should process all tokens again
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed


#
# Multimodal server (mmproj loaded) slot save/restore.
#
# A pure-text slot on a multimodal server and a slot containing images must both support save/restore.
# Erase remains gated on the slot's content.
#
# The media chunks travel in the state payload itself, via server_tokens::serialize().
# The save file trailer adds an encoder fingerprint, so a file written under a different
# mmproj - or different image token limits - is rejected instead of reusing KV that
# another encoder produced.
#

IMG_URL_CAT = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"
IMG_URL_TRUCK = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"


def _get_img_base64(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    return base64.b64encode(response.content).decode("utf-8")


@pytest.fixture
def mmproj_server():
    # tinygemma3 is a small multimodal model: the mmproj is provided by the HF registry API and auto-downloaded on first run.
    os.environ['LLAMA_MEDIA_MARKER'] = '<__media__>'
    mm_server = ServerPreset.tinygemma3()
    mm_server.slot_save_path = "./tmp"
    mm_server.temperature = 0.0
    return mm_server


def test_slot_save_restore_text_only_on_multimodal(mmproj_server):
    server = mmproj_server
    server.start()

    # A pure-text prompt processed on slot 1 of a multimodal server.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n = res.body["timings"]["prompt_n"]
    assert prompt_n > 0  # all tokens are processed

    # Saving a pure-text slot must succeed even though an mmproj is loaded.
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    assert n_saved > 0  # the slot KV (prompt + generated tokens) was written

    # Restore the saved state into slot 0; it must round-trip exactly.
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    # Prefix reuse is not checked with the default SWA cache.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_slot_save_restore_with_image(mmproj_server):
    server = mmproj_server
    # Use the full SWA cache so the restored image prefix can be reused.
    server.swa_full = True
    server.start()

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    content_cat = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert res.body["timings"]["cache_n"] == 0
    assert prompt_n_full > 32  # text plus image tokens are all processed

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    n_written = res.body["n_written"]
    assert n_saved > 0
    assert n_written > 0

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved
    assert res.body["n_read"] == n_written

    # a different image must not reuse the restored image tokens; only the text prefix before the image is common
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_TRUCK)],
        },
    })
    assert res.status_code == 200
    cache_n = res.body["timings"]["cache_n"]
    assert cache_n < 16
    assert res.body["timings"]["prompt_n"] == prompt_n_full - cache_n

    # restore again and resend the same image: the image tokens must be reused and greedy sampling must reproduce the original content
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["content"] == content_cat


def test_slot_save_restore_with_two_images(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.n_ctx = 2048  # two images need more than the default 512 per slot
    server.start()

    prompt = {
        "prompt_string": "A: <__media__> B: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT), _get_img_base64(IMG_URL_TRUCK)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert prompt_n_full > 64

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    content = res.body["content"]

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    content = res.body["content"]

    assert res.body["content"] == content


def test_slot_save_restore_with_image_across_restart(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.start()

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    content = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_restart.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    # restart the server with the same model and mmproj: the saved file must restore in the new process and the image KV must be reused
    server.stop()
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_restart.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["content"] == content


def test_slot_save_restore_image_payload_larger_than_context(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.start()

    # the slot context, as the server computed it (n_ctx split across the slots)
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    n_ctx_slot = res.body["default_generation_settings"]["n_ctx"]

    # a filler token, used to grow the prompt up to the slot context
    res = server.make_request("POST", "/tokenize", data={"content": " hello" * 8})
    assert res.status_code == 200
    assert len(res.body["tokens"]) == 8

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n" + " hello" * (n_ctx_slot - res.body["timings"]["prompt_n"] - 8),
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["cache_n"] + res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_large_payload.bin",
    })
    assert res.status_code == 200

    path = os.path.join(server.slot_save_path, "mm_slot_large_payload.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())
    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    assert payload_size > n_ctx_slot  # the scenario under test: the payload does not fit in n_ctx

    # drop the image from the slot, then restore it from the file
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_large_payload.bin",
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_restore_media_file_without_mmproj(mmproj_server):
    server = mmproj_server
    server.start()

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_no_mmproj.bin",
    })
    assert res.status_code == 200

    # restart the same model without the mmproj: restoring the media file must fail gracefully and leave the slot usable
    server.stop()
    server.no_mmproj = True
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_no_mmproj.bin",
    })
    assert res.status_code == 400
    assert "Cannot restore media tokens without an mmproj" in res.body["error"]["message"]

    # A failed restore must leave the slot empty and usable.
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
    content = res.body["content"]

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == 0
    assert res.body["content"] == content


#
# Slot state portability across --parallel. [TAG_STATE_SEQ_N_STREAM]
#
# The per-sequence state used to record the source instance's stream count and demand an exact
# match on restore. Since a non-unified KV cache has one stream per sequence, that count IS
# --parallel, so a state saved by a 1-slot server was rejected by a 4-slot one even when it fit
# comfortably - which is exactly the promotion path (a chat outgrowing a small high-concurrency
# entry) that the feature exists for.
#
# Note these all run non-unified (no --kv-unified), which is the only configuration in scope.
#

PROMPT_FR = "What is the capital of France?"
PROMPT_DE = "What is the capital of Germany?"


def _parallel_server(n_slots: int, n_ctx: int = 2048, **kwargs) -> ServerProcess:
    sp = ServerPreset.tinyllama2()
    sp.slot_save_path = "./tmp"
    sp.temperature = 0.0
    sp.n_slots = n_slots
    sp.n_ctx = n_ctx
    for k, v in kwargs.items():
        setattr(sp, k, v)
    return sp


def _seed_and_save(sp: ServerProcess, filename: str, prompt: str = PROMPT_FR, id_slot: int = 0) -> int:
    """Put a real conversation in a slot, save it, and return n_saved."""
    res = sp.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": id_slot,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = sp.make_request("POST", f"/slots/{id_slot}?action=save", data={"filename": filename})
    assert res.status_code == 200
    assert res.body["n_saved"] > 0
    return res.body["n_saved"]


@pytest.mark.parametrize("n_slots_src,n_slots_dst", [
    (1, 1),  # passed before the fix
    (1, 4),  # failed before the fix - the promotion that matters
    (4, 1),  # failed before the fix - the demotion
    (2, 4),  # neither side is 1, so the stream count really does differ
])
def test_slot_state_portable_across_parallel(n_slots_src, n_slots_dst):
    filename = f"xparallel_{n_slots_src}_{n_slots_dst}.bin"

    src = _parallel_server(n_slots_src)
    src.start()
    try:
        n_saved = _seed_and_save(src, filename)
    finally:
        src.stop()

    dst = _parallel_server(n_slots_dst)
    dst.start()
    try:
        res = dst.make_request("POST", "/slots/0?action=restore", data={"filename": filename})
        assert res.status_code == 200, res.body
        assert res.body["n_restored"] == n_saved
    finally:
        dst.stop()


def test_slot_state_portable_across_ctx_size():
    """--ctx-size was already portable; keep it that way while the framing changes."""
    filename = "xctx.bin"

    src = _parallel_server(1, n_ctx=1024)
    src.start()
    try:
        n_saved = _seed_and_save(src, filename)
    finally:
        src.stop()

    dst = _parallel_server(1, n_ctx=4096)
    dst.start()
    try:
        res = dst.make_request("POST", "/slots/0?action=restore", data={"filename": filename})
        assert res.status_code == 200, res.body
        assert res.body["n_restored"] == n_saved
    finally:
        dst.stop()


def test_restored_state_is_actually_used_across_parallel():
    """A 200 with zero cache hits would be worse than the old failure: we would stop
    reprocessing AND stop noticing. So check the restored KV really is reused."""
    filename = "xparallel_reuse.bin"

    src = _parallel_server(1)
    src.start()
    try:
        n_saved = _seed_and_save(src, filename, prompt=PROMPT_FR)
    finally:
        src.stop()

    dst = _parallel_server(4)
    dst.start()
    try:
        res = dst.make_request("POST", "/slots/0?action=restore", data={"filename": filename})
        assert res.status_code == 200, res.body
        assert res.body["n_restored"] == n_saved

        # continuing the same prompt must hit the restored cache. The two prompts share every
        # token but the country, so only the tail should be processed.
        res = dst.make_request("POST", "/completion", data={
            "prompt": PROMPT_DE,
            "id_slot": 0,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        prompt_n = res.body["timings"]["prompt_n"]
        assert prompt_n < 10, f"restored state was not reused, reprocessed {prompt_n} tokens"
    finally:
        dst.stop()


#
# The rejections that must survive the fix, each with its own error type so a caller can tell a
# file worth deleting from one that is simply wrong for this instance. [TAG_STATE_SEQ_STATUS]
#

def test_restore_rejects_state_larger_than_slot():
    filename = "xtoobig.bin"

    # one slot with the whole 2048-token context, filled well past 256 tokens
    src = _parallel_server(1, n_ctx=2048)
    src.start()
    try:
        long_prompt = "The quick brown fox jumps over the lazy dog. " * 60
        n_saved = _seed_and_save(src, filename, prompt=long_prompt)
        assert n_saved > 256
    finally:
        src.stop()

    # 8 slots over the same 2048 tokens leaves 256 per slot, so the state cannot fit
    dst = _parallel_server(8, n_ctx=2048)
    dst.start()
    try:
        res = dst.make_request("POST", "/slots/0?action=restore", data={"filename": filename})
        assert res.status_code == 400
        assert res.body["error"]["type"] == "slot_state_too_large_error", res.body
    finally:
        dst.stop()


def test_restore_rejects_different_kv_type():
    filename = "xkvtype.bin"

    src = _parallel_server(1, ctk="f16")
    src.start()
    try:
        _seed_and_save(src, filename)
    finally:
        src.stop()

    # a different K cache type changes the row size recorded per layer
    dst = _parallel_server(1, ctk="q8_0")
    dst.start()
    try:
        res = dst.make_request("POST", "/slots/0?action=restore", data={"filename": filename})
        assert res.status_code == 400
        assert res.body["error"]["type"] == "slot_state_incompatible_error", res.body
    finally:
        dst.stop()


def test_restore_rejects_truncated_file():
    filename = "xtruncated.bin"

    sp = _parallel_server(1)
    sp.start()
    try:
        _seed_and_save(sp, filename)
    finally:
        sp.stop()

    # lop off the tail - the header still parses, the state does not
    path = os.path.join(TMP_DIR, filename)
    size = os.path.getsize(path)
    with open(path, "r+b") as f:
        f.truncate(size // 2)

    sp = _parallel_server(1)
    sp.start()
    try:
        res = sp.make_request("POST", "/slots/0?action=restore", data={"filename": filename})
        assert res.status_code == 400
        assert res.body["error"]["type"] == "slot_state_corrupt_error", res.body
    finally:
        sp.stop()


def test_restore_rejects_missing_file():
    sp = _parallel_server(1)
    sp.start()
    try:
        res = sp.make_request("POST", "/slots/0?action=restore", data={"filename": "does_not_exist.bin"})
        assert res.status_code == 400
        assert res.body["error"]["type"] == "slot_state_corrupt_error", res.body
    finally:
        sp.stop()


#
# Slot state I/O runs off the inference thread. [TAG_SLOT_IO_ASYNC]
#
# This asserts the functional half only - that other slots keep serving correct results while a
# save and a restore are in flight, and that the transferred state still lands intact. The
# latency half of the property needs a real model on a real --slot-save-path, because a
# stories260K state is far too small for the old inline write to show up as a stall.
#

def test_slot_io_does_not_disturb_other_slots():
    from concurrent.futures import ThreadPoolExecutor

    filename = "xconcurrent.bin"

    sp = _parallel_server(4)
    sp.start()
    try:
        n_saved = _seed_and_save(sp, filename, id_slot=0)

        def busy(id_slot):
            res = sp.make_request("POST", "/completion", data={
                "prompt": PROMPT_DE,
                "id_slot": id_slot,
                "n_predict": 32,
                "cache_prompt": True,
            })
            return res

        with ThreadPoolExecutor(max_workers=4) as ex:
            others = [ex.submit(busy, i) for i in (1, 2, 3)]
            io = ex.submit(lambda: sp.make_request(
                "POST", "/slots/0?action=restore", data={"filename": filename}))

            for f in others:
                r = f.result()
                assert r.status_code == 200, r.body
                assert len(r.body["content"]) > 0

            r = io.result()
            assert r.status_code == 200, r.body
            assert r.body["n_restored"] == n_saved
    finally:
        sp.stop()


def test_slot_erase_text_only_on_multimodal(mmproj_server):
    server = mmproj_server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n = res.body["timings"]["prompt_n"]
    assert prompt_n > 0  # all tokens are processed

    # Erasing a pure-text slot must succeed even though an mmproj is loaded.
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    # Re-running the same prompt should process all tokens again.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == prompt_n  # all tokens are processed again


def test_slot_restore_media_file_from_another_encoder(mmproj_server):
    server = mmproj_server
    server.start()

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_fingerprint.bin",
    })
    assert res.status_code == 200

    # the image token limits change how many tokens an image expands to, so the KV in the file
    # no longer describes what this server would produce. the chunk ids hash the media bytes
    # alone and would still match, so only the fingerprint can catch this.
    server.stop()
    server.image_min_tokens = 64
    server.image_max_tokens = 64
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_fingerprint.bin",
    })
    assert res.status_code == 400
    assert "different multimodal projector" in res.body["error"]["message"]

    # a rejected restore must leave the slot usable
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
