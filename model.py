"""Qwen2-Audio QLoRA backend, shared by every training and evaluation method."""
import numpy as np
import torch


class AudioLanguageModel:
    def __init__(self, settings, model_path=None, training=True):
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration
        from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Training/evaluation require an NVIDIA CUDA GPU with bfloat16 support')
        self.settings = settings
        source = model_path or settings.model_id
        options = {'local_files_only': True} if model_path else {'revision': settings.revision}
        self.processor = AutoProcessor.from_pretrained(source, **options)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(source,
            quantization_config=quant, dtype=torch.bfloat16, device_map={'': 0}, **options)
        self.model.config.use_cache = False
        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=training)
        targets = []
        for name, module in self.model.named_modules():
            language = 'language_model' in name and name.rsplit('.', 1)[-1] in {'q_proj', 'v_proj'}
            projector = any(part in name.split('.') for part in
                            ['multi_modal_projector', 'multimodal_projector', 'audio_projector'])
            linear = module.__class__.__name__ in {'Linear', 'Linear4bit', 'Linear8bitLt'}
            if language or (projector and linear):
                targets.append(name)
        if len(targets) != 65:
            raise ValueError(f'Expected 65 adapter target modules; found {len(targets)}')
        self.model.add_adapter(LoraConfig(task_type=TaskType.CAUSAL_LM, r=settings.lora_rank,
            lora_alpha=settings.lora_alpha, lora_dropout=settings.lora_dropout,
            bias='none', target_modules=sorted(targets)), adapter_name='adapter')
        self.model.set_adapter('adapter')
        self.tower = next(m for m in self.model.modules() if m.__class__.__name__ == 'Qwen2AudioEncoder')
        self.tower.requires_grad_(False)
        self.train(training)
        if len(self.parameters()) != 130:
            raise ValueError('Expected only 130 trainable LoRA A/B tensors')

    def parameters(self):
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def state(self):
        return {n: p.detach().cpu().clone() for n, p in self.parameters()}

    def load(self, state):
        parameters = dict(self.parameters())
        if parameters.keys() != state.keys():
            raise ValueError('Adapter keys differ from the initialized model')
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = state[name]
                if value.shape != parameter.shape or not torch.isfinite(value).all():
                    raise ValueError('Invalid adapter tensor')
                parameter.copy_(value.to(parameter.device, dtype=parameter.dtype))

    def train(self, mode=True):
        self.model.train(mode)
        self.tower.eval()

    def loss(self, item, target=None):
        prompt = [dict(role='system', content='Below is a conversation between an interviewer '
            'and a participant. Please analyze the transcripts and audio, and find whether '
            'the participant is affected by depression.'),
            dict(role='user', content=[dict(type='audio', audio_url='local://participant_audio.wav'),
                dict(type='text', text='Interview conversation:\n'+item['conversation_text']+'\nResponse:')])]
        prefix = self.processor.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        full = self.processor.apply_chat_template(prompt+[dict(role='assistant',
            content=target or item['target_text'])], add_generation_prompt=False, tokenize=False)
        tokenizer = self.processor.tokenizer
        prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
        full_ids = tokenizer(full, add_special_tokens=False)['input_ids']
        if full_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError('Chat-template prefix is not an exact token prefix')
        count = len(full_ids)-len(prefix_ids)
        if count <= 0:
            raise ValueError('No supervised assistant tokens')
        # Truncate the waveform identically for both candidate targets.
        audio = np.asarray(item['audio'], dtype=np.float32)[:self.settings.audio_limit_seconds*16000]
        inputs = self.processor(text=full, audio=[audio], sampling_rate=16000,
                                return_tensors='pt', padding=True)
        device = next(iter(self.parameters()))[1].device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs['input_ids'].clone()
        labels[:, :-count] = -100
        # Qwen returns the mean answer-token NLL, including the assistant closing tokens.
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            return self.model(**inputs, labels=labels).loss

    @torch.inference_mode()
    def score(self, item):
        self.train(False)
        depressed = float(self.loss(item, 'Depressed'))
        not_depressed = float(self.loss(item, 'Not Depressed'))
        return not_depressed-depressed
