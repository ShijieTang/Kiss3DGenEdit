from pipeline.kiss3d_wrapper import init_wrapper_from_config, run_edit_3d_bundle
import os
from pipeline.utils import logger, TMP_DIR, OUT_DIR
import time
if __name__ == "__main__":
    os.makedirs(os.path.join(OUT_DIR, 'text_to_3d'), exist_ok=True)
    k3d_wrapper = init_wrapper_from_config('./pipeline/pipeline_config/default.yaml')
    src_prompt = 'A doll of a girl in Harry Potter'
    tgt_prompt = 'A doll of a girl in Harry Potter with glasses'
    name = src_prompt.replace(' ', '_') + '_to_' + tgt_prompt.replace(' ', '_')
    os.system(f'rm -rf {TMP_DIR}/*')
    end = time.time()
    src_img, tgt_img, _, _ = run_edit_3d_bundle(k3d_wrapper, prompt_src=src_prompt, prompt_tgt=tgt_prompt)
    print(f" edit_3d_bundle time: {time.time() - end}")
    os.system(f'cp -f {src_img} {OUT_DIR}/text_to_3d/{name}_src_3d_bundle.png')
    os.system(f'cp -f {tgt_img} {OUT_DIR}/text_to_3d/{name}_tgt_3d_bundle.png')
    

