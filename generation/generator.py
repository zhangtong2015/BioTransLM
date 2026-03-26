# 模块M：生成器 (BioGenerator)
# 负责人：AI 2
# 输入：prompt: str, control: Dict
# 输出：{'generated_text': str}
#
# 已重构：完全使用生物启发机制替换Transformer解码器
# - 序列预测：基于HTM时序记忆
# - 词汇映射：SDR稀疏表示
# - 无自注意力、无Transformer组件

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)

from core.base_module import BaseModule
from config import BaseConfig
from utils.common_utils import utils
from generation.sequence_predictor import SequencePredictorTM, SequencePredictorConfig
from generation.sdr_vocabulary import SDRVocabulary, SDRVocabularyConfig

# 尝试导入transformers，否则使用简化实现
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class GeneratorConfig(BaseConfig):
    """生成器配置 - 完全生物启发版"""
    hidden_dim: int = 768
    vocab_size: int = 50257  # GPT-2词汇表大小
    max_sequence_length: int = 512
    dropout_rate: float = 0.1
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    tokenizer_name: str = "gpt2"
    use_pretrained: bool = False
    max_new_tokens: int = 100
    pad_token_id: int = 50256
    eos_token_id: int = 50256
    bos_token_id: int = 50256
    
    # HTM配置
    n_columns: int = 4096
    n_cells_per_col: int = 16
    segment_threshold: int = 12
    active_column_rate: float = 0.02
    
    # SDR配置
    sdr_size: int = 4096
    sdr_sparsity: float = 0.02
    
    # 上下文配置
    use_context_modulation: bool = True
    context_gate_dim: int = 768


class BioGenerator(BaseModule):
    """
    生物启发式生成器模块
    
    完全替代Transformer解码器的纯生物启发架构：
    1. 序列预测：基于HTM时序记忆
    2. 词汇映射：SDR稀疏表示双向映射
    3. 上下文调制：神经调节信号
    4. 无任何Transformer组件
    
    核心数据流：
    [输入token/embeds] → [SDR编码] → [序列预测TM] → [上下文调制] → [SDR解码] → [token输出]
    """
    
    def __init__(
        self, 
        config: Optional[GeneratorConfig] = None,
        module_name: str = "bio_generator"
    ):
        config = config or GeneratorConfig()
        super().__init__(config=config, module_name=module_name)
    
    def _initialize_module(self) -> None:
        """初始化生物启发式生成器组件"""
        # 初始化tokenizer（兼容现有接口）
        self._initialize_tokenizer()
        
        # 1. SDR词汇表：token ↔ 稀疏表示 双向映射
        sdr_vocab_config = SDRVocabularyConfig(
            vocab_size=self.config.vocab_size,
            sdr_size=self.config.sdr_size,
            activation_sparsity=self.config.sdr_sparsity,
            embedding_dim=self.config.hidden_dim
        )
        self.sdr_vocabulary = SDRVocabulary(sdr_vocab_config)
        
        # 2. 序列预测器：基于HTM时序记忆
        seq_pred_config = SequencePredictorConfig(
            n_columns=self.config.n_columns,
            n_cells_per_col=self.config.n_cells_per_col,
            output_dim=self.config.hidden_dim,
            vocab_size=self.config.vocab_size,
            segment_threshold=self.config.segment_threshold,
            active_column_rate=self.config.active_column_rate
        )
        self.sequence_predictor = SequencePredictorTM(seq_pred_config)
        
        # 3. 嵌入到SDR的投影层
        self.embedding_to_sdr = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim * 2, self.config.sdr_size),
            nn.ReLU()
        )
        
        # 4. SDR到列激活的投影
        self.sdr_to_columns = nn.Sequential(
            nn.Linear(self.config.sdr_size, self.config.n_columns),
            nn.ReLU()
        )
        
        # 5. 上下文融合门控
        if self.config.use_context_modulation:
            self.context_gate = nn.Sequential(
                nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
                nn.LayerNorm(self.config.hidden_dim),
                nn.Sigmoid()
            )
        
        # 6. 重复抑制机制
        self.repetition_suppression = nn.Parameter(
            torch.ones(1, self.config.vocab_size),
            requires_grad=False
        )
        
        # 7. Dropout层
        self.dropout = nn.Dropout(self.config.dropout_rate)
        
        logger.info(f"生物启发式生成器初始化完成: "
                        f"SDR维度={self.config.sdr_size}, "
                        f"HTM列数={self.config.n_columns}, "
                        f"稀疏度={self.config.sdr_sparsity:.1%}")
    
    def _initialize_tokenizer(self) -> None:
        """初始化tokenizer"""
        if TRANSFORMERS_AVAILABLE and self.config.use_pretrained:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.config.vocab_size = self.tokenizer.vocab_size
                self.config.pad_token_id = self.tokenizer.pad_token_id
                self.config.eos_token_id = self.tokenizer.eos_token_id
                self.config.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
                self._tokenizer_available = True
            except Exception as e:
                print(f"加载tokenizer失败，使用简化模式: {e}")
                self._tokenizer_available = False
                self.tokenizer = None
        else:
            self._tokenizer_available = False
            self.tokenizer = None
    
    def _tokenize(self, text: Union[str, List[str]]) -> torch.Tensor:
        """文本转token ID"""
        if isinstance(text, str):
            text = [text]
        
        if self._tokenizer_available:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.config.max_sequence_length,
                return_tensors="pt"
            )
            return inputs['input_ids'].to(self._device)
        else:
            # 简化模式：随机生成token ids用于测试
            batch_size = len(text)
            seq_len = min(32, max(len(t.split()) for t in text) + 5)
            return torch.randint(
                0, self.config.vocab_size, 
                (batch_size, seq_len)
            ).to(self._device)
    
    def _ids_to_text(self, token_ids: torch.Tensor) -> List[str]:
        """token ID转文本"""
        if not self._tokenizer_available:
            return [f"[generated_sequence_{i}]" for i in range(token_ids.shape[0])]
        
        texts = []
        for ids in token_ids.cpu().numpy():
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)
        return texts
    
    def _apply_repetition_penalty(self, 
                                  logits: torch.Tensor, 
                                  generated_tokens: torch.Tensor,
                                  penalty: float = 1.1) -> torch.Tensor:
        """应用重复惩罚"""
        if penalty <= 1.0 or generated_tokens.numel() == 0:
            return logits
        
        # 收集已生成的token
        generated_flat = generated_tokens.view(-1).unique()
        
        # 对已生成的token应用惩罚
        penalty_mask = torch.ones_like(logits)
        penalty_mask[:, generated_flat] = penalty
        
        return logits / penalty_mask
    
    def _process_input_to_columns(self,
                                 input_ids: Optional[torch.Tensor] = None,
                                 input_embeds: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将输入转换为HTM列激活表示
        
        两种输入模式：
        1. input_ids: token ID → SDR → 列激活
        2. input_embeds: 嵌入 → 投影到SDR → 列激活
        """
        batch_size = input_ids.shape[0] if input_ids is not None else input_embeds.shape[0]
        seq_len = input_ids.shape[1] if input_ids is not None else input_embeds.shape[1]
        
        if input_ids is not None:
            # 模式1：从token ids转换为SDR（通过forward方法获取字典返回值）
            sdr_result = self.sdr_vocabulary(input_ids, mode='token_to_sdr')
            sdr = sdr_result['sdr']  # [B, T, sdr_size]
            embeddings = sdr_result['embeddings']
        else:
            # 模式2：从嵌入投影到SDR
            embeddings = input_embeds
            sdr_flat = self.embedding_to_sdr(input_embeds.reshape(-1, self.config.hidden_dim))
            sdr = sdr_flat.reshape(batch_size, seq_len, self.config.sdr_size)
        
        # 将SDR转换为列激活
        columns_flat = self.sdr_to_columns(sdr.reshape(-1, self.config.sdr_size))
        column_activations = columns_flat.reshape(batch_size, seq_len, self.config.n_columns)
        
        return column_activations, embeddings
    
    def forward(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        control: Optional[Dict[str, Any]] = None,
        input_embeds: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: bool = True,
        style_embeds: Optional[torch.Tensor] = None,
        return_probs: bool = False,
        reset_state: bool = True
    ) -> Dict[str, Any]:
        """
        前向传播 - 生物启发式文本生成
        
        核心流程（无Transformer）：
        1. 输入编码 → SDR稀疏表示
        2. HTM时序记忆处理 → 序列状态
        3. 上下文调制 → 门控融合
        4. SDR解码 → token概率分布
        5. 自回归生成（可选）
        """
        # 从control字典中提取参数
        control = control or {}
        
        # 处理max_new_tokens：明确传入0时使用0，否则使用传入值或默认值
        if 'max_new_tokens' in control:
            max_new_tokens = control['max_new_tokens']
        elif max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        
        temperature = control.get('temperature', temperature or self.config.temperature)
        top_k = control.get('top_k', top_k or self.config.top_k)
        top_p = control.get('top_p', top_p or self.config.top_p)
        repetition_penalty = control.get('repetition_penalty', repetition_penalty or self.config.repetition_penalty)
        do_sample = control.get('do_sample', do_sample)
        
        max_length = max_length or self.config.max_sequence_length
        
        # 重置预测器状态（默认行为）
        if reset_state:
            self.sequence_predictor.reset_state()
        
        # 准备输入：优先使用prompt，其次input_ids，最后input_embeds
        if prompt is not None:
            input_ids = self._tokenize(prompt)
        elif input_ids is None and input_embeds is None:
            raise ValueError("必须提供prompt、input_ids或input_embeds之一")
        
        # 确保在正确的设备上
        if input_ids is not None:
            input_ids = input_ids.to(self._device)
        if input_embeds is not None:
            input_embeds = input_embeds.to(self._device)
        
        batch_size = input_ids.shape[0] if input_ids is not None else input_embeds.shape[0]
        
        # ========== 阶段1：输入转换为SDR表示 ==========
        column_activations, embeddings = self._process_input_to_columns(input_ids, input_embeds)
        
        # ========== 阶段2：上下文融合（如果有） ==========
        context_signal = None
        if context and 'context_embeds' in context:
            context_embeds = context['context_embeds']
            if self.config.use_context_modulation:
                if len(context_embeds.shape) == 2:
                    context_embeds = context_embeds.unsqueeze(1)
                context_signal = context_embeds.expand(-1, column_activations.shape[1], -1)
        
        # ========== 阶段3：HTM序列预测 ==========
        seq_result = self.sequence_predictor(
            column_activations=column_activations,
            context=context_signal,
            learn=False  # 推理阶段默认不学习
        )
        
        sequence_output = seq_result['sequence_output']  # [B, T, D]
        
        # ========== 阶段4：风格/上下文调制（可选） ==========
        if context_signal is not None and self.config.use_context_modulation:
            gate_input = torch.cat([sequence_output, context_signal], dim=-1)
            gate = self.context_gate(gate_input)
            sequence_output = sequence_output * gate + sequence_output * 0.1
        
        if style_embeds is not None:
            style_weights = torch.sigmoid(style_embeds.mean(dim=1, keepdim=True))
            sequence_output = sequence_output * style_weights
        
        # ========== 阶段5：转换为token概率 ==========
        # 将输出投影到SDR空间
        output_sdr_flat = self.embedding_to_sdr(sequence_output.reshape(-1, self.config.hidden_dim))
        output_sdr = output_sdr_flat.reshape(batch_size, sequence_output.shape[1], self.config.sdr_size)
        
        # SDR → token logits
        token_result = self.sdr_vocabulary(
            output_sdr,
            mode='sdr_to_token',
            temperature=temperature
        )
        logits = token_result['token_logits']  # [B, T, V]
        
        # ========== 阶段6：自回归生成（可选） ==========
        generation_result = {
            'generated_ids': input_ids if input_ids is not None else torch.argmax(logits, dim=-1),
            'generated_text': None,
            'complete': False
        }
        
        if max_new_tokens > 0:
            generation_result = self._autoregressive_generation(
                initial_ids=input_ids if input_ids is not None else torch.argmax(logits, dim=-1),
                initial_state=seq_result,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                context=context
            )
        
        # 转换为文本
        generation_result['generated_text'] = self._ids_to_text(generation_result['generated_ids'])
        
        # ========== 构建结果 ==========
        result = {
            'generated_text': generation_result['generated_text'],
            'generated_ids': generation_result['generated_ids'],
            'generation_state': {
                'sequence_output': sequence_output,
                'output_sdr': output_sdr,
                'logits': logits,
                'token_probs': token_result['token_probs'],
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p
            }
        }
        
        if return_probs:
            result['token_probabilities'] = token_result['token_probs']
        
        return result
    
    def _autoregressive_generation(
        self,
        initial_ids: torch.Tensor,
        initial_state: Dict[str, Any],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        do_sample: bool,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        自回归生成新token（生物启发版）
        
        每一步：
        1. 获取最后一列激活 → SDR表示
        2. 序列预测器预测下一状态
        3. 转换为token概率
        4. 采样新token
        5. 更新状态历史
        """
        batch_size = initial_ids.shape[0]
        # 确保在正确的设备上
        initial_ids = initial_ids.to(self._device)
        device = self._device
        
        generated_tokens = []
        current_ids = initial_ids
        all_logits = []
        
        # 重置抑制状态
        suppression_state = self.repetition_suppression.repeat(batch_size, 1).to(device)
        
        for step in range(max_new_tokens):
            # 1. 处理最新输入
            with torch.no_grad():
                last_ids = current_ids[:, -1:]  # [B, 1]
                column_activations, _ = self._process_input_to_columns(input_ids=last_ids)
                
                # 2. HTM预测下一步
                context_signal = None
                if context and 'context_embeds' in context:
                    context_signal = context['context_embeds'][:, None, :]  # [B, 1, D]
                
                seq_result = self.sequence_predictor(
                    column_activations=column_activations,
                    context=context_signal,
                    learn=False
                )
                
                sequence_output = seq_result['sequence_output']  # [B, 1, D]
                
                # 3. 转换为token概率
                output_sdr = self.embedding_to_sdr(sequence_output.squeeze(1))  # [B, S]
                token_logits = self.sdr_vocabulary.sdr_to_token_logits(
                    output_sdr.unsqueeze(1),
                    temperature=temperature
                )  # [B, 1, V]
                
                token_logits = token_logits.squeeze(1)  # [B, V]
                all_logits.append(token_logits.unsqueeze(1))
            
            # 4. 应用重复惩罚
            if repetition_penalty > 1.0 and len(generated_tokens) > 0:
                generated_history = torch.cat([initial_ids] + generated_tokens, dim=1)
                token_logits = self._apply_repetition_penalty(
                    token_logits, 
                    generated_history,
                    repetition_penalty
                )
            
            # 5. 采样新token
            if do_sample:
                # Top-k采样
                if top_k > 0:
                    indices_to_remove = token_logits < torch.topk(token_logits, top_k)[0][:, -1, None]
                    token_logits[indices_to_remove] = -float('inf')
                
                # Top-p (nucleus) 采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits / temperature, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    token_logits[indices_to_remove] = -float('inf')
                
                probs = F.softmax(token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            else:
                # 贪婪解码
                next_token = torch.argmax(token_logits, dim=-1, keepdim=True)  # [B, 1]
            
            # 6. 收集结果
            generated_tokens.append(next_token)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # 7. 检测EOS（如果有可用tokenizer）
            if self._tokenizer_available:
                eos_count = (next_token == self.config.eos_token_id).sum()
                if eos_count == batch_size:
                    break
        
        # 整理结果
        if len(generated_tokens) > 0:
            full_sequence = torch.cat([initial_ids] + generated_tokens, dim=1)
            all_logits = torch.cat(all_logits, dim=1)
        else:
            full_sequence = initial_ids
            all_logits = None
        
        return {
            'generated_ids': full_sequence,
            'logits': all_logits,
            'complete': True
        }
    
    def reset_generation_state(self):
        """重置生成状态，用于新的生成会话"""
        self.sequence_predictor.reset_state()


# 为保持向后兼容性，保留原类名别名
ResponseGenerator = BioGenerator
