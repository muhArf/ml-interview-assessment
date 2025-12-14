# utils/memory_manager.py
import gc
import torch
import streamlit as st

def clear_memory():
    """Clear memory untuk mencegah OOM di Streamlit Cloud"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clear session state jika terlalu besar
    if 'responses' in st.session_state:
        # Hapus audio paths dari memory
        for q_id, response in st.session_state.responses.items():
            if 'audio_path' in response:
                try:
                    import os
                    if os.path.exists(response['audio_path']):
                        os.remove(response['audio_path'])
                except:
                    pass
    
    return True