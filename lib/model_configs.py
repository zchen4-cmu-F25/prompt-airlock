# MARK: prompt-airlock — local Mistral weights + semantic_smoothing.py entry point
MODELS = {
    'mistral': {
        'model_path': r'D:\models\Mistral-7B-Instruct-v0.2',
        'tokenizer_path': r'D:\models\Mistral-7B-Instruct-v0.2',
        'conversation_template': 'mistral',
    },
    'llama2': {
        'model_path': '/shared_data0/arobey1/llama-2-7b-chat-hf',
        'tokenizer_path': '/shared_data0/arobey1/llama-2-7b-chat-hf',
        'conversation_template': 'llama-2'
    },
    'vicuna': {
        'model_path': '/shared_data0/arobey1/vicuna-13b-v1.5',
        'tokenizer_path': '/shared_data0/arobey1/vicuna-13b-v1.5',
        'conversation_template': 'vicuna'
    }
}