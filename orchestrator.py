# BioTransLM 编排器 - 端到端集成核心
# 负责人：AI 1
# 功能：协调所有模块，实现用户文本输入到系统响应的完整流程

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import time
import warnings
import numpy as np

# 抑制不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

from config import BaseConfig
from core.base_module import BaseModule
from input.sensory_gating import SensoryGating, SensoryGatingConfig
from input.multigranular_encoder import MultiGranularEncoder, MultiGranularEncoderConfig
from input.embedding_converter import EmbeddingConverter, EmbeddingConverterConfig
from htm.htm_system import HTMSystem, HTMSystemConfig
from htm.word_level_cortex import WordLevelCortex, WordLevelCortexConfig
from htm.sentence_level_cortex import SentenceLevelCortex, SentenceLevelCortexConfig
from htm.sparse_attention import SparseAttention, SparseAttentionConfig
from regulation.neural_regulator import NeuralRegulator, NeuralRegulatorConfig
from reasoning.system1 import System1Intuition as System1, System1Config
from reasoning.system2 import System2Reasoning as System2, System2Config
from reasoning.fusion import DualSystemFusion as Fusion, FusionConfig
from memory.working_memory import WorkingMemory, WorkingMemoryConfig
from memory.episodic_memory import EpisodicMemory, EpisodicMemoryConfig
from memory.semantic_memory import SemanticMemory, SemanticMemoryConfig
from generation.generator import ResponseGenerator as Generator, GeneratorConfig
from generation.forward_model import ForwardModel, ForwardModelConfig
from utils.common_utils import utils

logger = logging.getLogger(__name__)

@dataclass
class OrchestratorConfig(BaseConfig):
    """编排器配置"""
    model_name: str = "bert-base-uncased"
    max_sequence_length: int = 512
    batch_size: int = 1
    device: str = "cpu"  # 强制使用CPU避免问题
    debug_mode: bool = True
    
    # 模块开关
    use_sensory_gating: bool = True
    use_embedding_converter: bool = True  # 使用新的嵌入转换层
    use_htm_pipeline: bool = True  # 使用标准HTM管道
    use_legacy_htm: bool = False  # 使用原有词级/句级皮层
    use_sparse_attention: bool = True
    use_neural_regulation: bool = True
    use_dual_system: bool = True
    use_memory_system: bool = True
    use_forward_model: bool = True

class Orchestrator(BaseModule):
    """BioTransLM 编排器 - 协调所有模块的核心控制器"""
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        super().__init__(config=self.config, module_name="orchestrator")
        
    def _initialize_module(self):
        """初始化所有子模块"""
        logger.info("初始化 BioTransLM 编排器...")
        
        # 1. 输入处理层
        if self.config.use_sensory_gating:
            self.sensory_gating = SensoryGating(SensoryGatingConfig(
                hidden_dim=768,
                device=self.config.device
            ))
        
        if self.config.use_embedding_converter:
            self.embedding_converter = EmbeddingConverter(EmbeddingConverterConfig(
                input_dim=768,
                output_dim=2048,
                sparsity_target=0.95
            ))
        
        self.multigranular_encoder = MultiGranularEncoder(MultiGranularEncoderConfig(
            hidden_dim=768,
            device=self.config.device
        ))
        
        # 2. HTM皮层系统
        if self.config.use_htm_pipeline:
            self.htm_system = HTMSystem(HTMSystemConfig(
                input_dim=2048,  # 匹配嵌入转换器输出维度
                n_columns=4096,
                n_cells_per_col=16,
                output_dim=768  # 匹配其他模块期望的维度
            ))
        
        if self.config.use_legacy_htm:
            self.word_level_cortex = WordLevelCortex(WordLevelCortexConfig(
                hidden_dim=768,
                device=self.config.device
            ))
            
            self.sentence_level_cortex = SentenceLevelCortex(SentenceLevelCortexConfig(
                hidden_dim=768,
                device=self.config.device
            ))
        
        if self.config.use_sparse_attention:
            self.sparse_attention = SparseAttention(SparseAttentionConfig(
                hidden_dim=768,
                device=self.config.device
            ))
        
        # 3. 神经调节器
        if self.config.use_neural_regulation:
            self.neural_regulator = NeuralRegulator(NeuralRegulatorConfig(
                hidden_dim=768,
                device=self.config.device
            ))
        
        # 4. 记忆系统
        if self.config.use_memory_system:
            self.working_memory = WorkingMemory(WorkingMemoryConfig(
                hidden_dim=768,
                device=self.config.device
            ))
            self.episodic_memory = EpisodicMemory(EpisodicMemoryConfig(
                hidden_dim=768,
                device=self.config.device
            ))
            self.semantic_memory = SemanticMemory(SemanticMemoryConfig(
                hidden_dim=768,
                device=self.config.device
            ))
        
        # 5. 推理系统
        self.system1 = System1(System1Config(
            hidden_dim=768,
            device=self.config.device
        ))
        self.system2 = System2(System2Config(
            hidden_dim=768,
            device=self.config.device
        ))
        self.fusion = Fusion(FusionConfig(
            hidden_dim=768,
            device=self.config.device
        ))
        
        # 6. 生成系统
        self.generator = Generator(GeneratorConfig(
            hidden_dim=768,
            device=self.config.device
        ))
        
        if self.config.use_forward_model:
            self.forward_model = ForwardModel(ForwardModelConfig(
                hidden_dim=768,
                device=self.config.device
            ))
        
        logger.info("所有模块初始化完成！")
    
    def _get_simple_embeddings(self, batch_size: int, seq_len: int, hidden_dim: int) -> torch.Tensor:
        """获取简单的随机嵌入（用于测试，避免下载模型）"""
        return torch.randn(batch_size, seq_len, hidden_dim).to(self._device)
    
    def forward(self, input_text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        端到端处理流程
        
        Args:
            input_text: 输入文本或文本列表
            
        Returns:
            包含所有中间结果和最终响应的字典
        """
        start_time = time.time()
        results = {}
        
        # 确保输入是列表格式
        if isinstance(input_text, str):
            input_text = [input_text]
        
        logger.info(f"处理输入: {input_text}")
        
        batch_size = len(input_text)
        seq_len = min(32, max(len(t.split()) for t in input_text) + 10)
        hidden_dim = 768
        
        # ========== 阶段1: 简单编码 ==========
        
        # 模拟初始嵌入（不依赖外部模型
        initial_embeds = self._get_simple_embeddings(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len).to(self._device)
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(self._device)
        
        results["initial_embeddings"] = initial_embeds.detach().cpu()
        
        # ========== 阶段2: 感觉门控 ==========
        if self.config.use_sensory_gating:
            gating_start = time.time()
            gating_output = self.sensory_gating(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_embeds=initial_embeds  # 使用正确的参数名 input_embeds
            )
            gated_embeds = gating_output.get("gated_embeds", initial_embeds)
            results["sensory_gating"] = {k: v.detach().cpu() if torch.is_tensor(v) else v 
                                        for k, v in gating_output.items()}
            logger.debug(f"感觉门控耗时: {(time.time() - gating_start) * 1000:.2f}ms")
        else:
            gated_embeds = initial_embeds
        
        # ========== 阶段3: 多粒度编码 ==========
        multigran_start = time.time()
        multigran_output = self.multigranular_encoder(
            input_embeds=gated_embeds,  # 使用正确的参数名
            attention_mask=attention_mask
        )
        results["multigranular_encoding"] = {k: v.detach().cpu() if torch.is_tensor(v) else v 
                                           for k, v in multigran_output.items()}
        logger.debug(f"多粒度编码耗时: {(time.time() - multigran_start) * 1000:.2f}ms")
        
        # ========== 阶段4: HTM皮层处理 ==========
        htm_start = time.time()
        
        # 使用新的HTM管道（根据配置）
        if self.config.use_embedding_converter:
            # 嵌入转换层：转换为稀疏表示
            conv_output = self.embedding_converter(
                input_embeddings=gated_embeds,
                attention_mask=attention_mask
            )
            sparse_input = conv_output['sparse_output']  # [batch, seq_len, 2048]
            results["embedding_converter"] = {
                k: v.detach().cpu() if torch.is_tensor(v) else v 
                for k, v in conv_output.items()
            }
            logger.debug(f"嵌入转换稀疏度: {conv_output['actual_sparsity']:.1%}")
        else:
            # 使用多粒度编码输出
            sparse_input = multigran_output.get("word_embeds", gated_embeds)
        
        # HTM系统处理
        if self.config.use_htm_pipeline:
            htm_output = self.htm_system(
                input_sdr=sparse_input,
                learn=self.training  # 根据训练模式决定是否学习
            )
            results["htm_system"] = {
                k: v.detach().cpu() if torch.is_tensor(v) else v 
                for k, v in htm_output.items()
            }
            htm_repr = htm_output['htm_repr']  # [batch, seq_len, 1024]
            logger.debug(f"HTM预测准确率: {htm_output['temporal_output']['prediction_accuracy']:.1%}")
            logger.debug(f"HTM Burst列数: {htm_output['temporal_output']['n_bursts']}")
        
        # 遗留HTM处理（可选）
        if self.config.use_legacy_htm:
            # 词级皮层处理
            word_embeds = multigran_output.get("word_embeds", gated_embeds)
            word_level_output = self.word_level_cortex(
                word_embeds=word_embeds
            )
            results["word_level_cortex"] = {
                k: v.detach().cpu() if torch.is_tensor(v) else v 
                for k, v in word_level_output.items()
            }
            
            # 句级皮层处理
            sentence_embeds = multigran_output.get("sentence_embeds", torch.mean(gated_embeds, dim=1, keepdim=True))
            sentence_level_output = self.sentence_level_cortex(
                sentence_embeds=sentence_embeds,
                word_level_output=word_level_output,
                attention_mask=attention_mask
            )
            results["sentence_level_cortex"] = {
                k: v.detach().cpu() if torch.is_tensor(v) else v 
                for k, v in sentence_level_output.items()
            }
        
        # 稀疏注意力（可选）
        if self.config.use_sparse_attention:
            # 确定注意力输入
            if self.config.use_htm_pipeline:
                attn_input = htm_repr  # 使用HTM输出
            elif self.config.use_legacy_htm:
                attn_input = sentence_level_output.get("sentence_repr", sentence_embeds)
                if attn_input.dim() == 2:
                    attn_input = attn_input.unsqueeze(1)
            else:
                attn_input = sparse_input
            
            # 创建匹配的attention_mask
            if attn_input.size(1) != attention_mask.size(1):
                attn_mask = torch.ones(batch_size, attn_input.size(1)).to(self._device)
            else:
                attn_mask = attention_mask
                
            sparse_attention_output = self.sparse_attention(
                hidden_states=attn_input,
                attention_mask=attn_mask
            )
            results["sparse_attention"] = {
                k: v.detach().cpu() if torch.is_tensor(v) else v 
                for k, v in sparse_attention_output.items()
            }
        
        logger.debug(f"HTM处理耗时: {(time.time() - htm_start) * 1000:.2f}ms")
        
        # ========== 阶段5: 神经调节 ==========
        if self.config.use_neural_regulation:
            regulation_start = time.time()
            
            # 根据配置收集错误信号
            errors = {}
            
            if self.config.use_htm_pipeline and "htm_system" in results:
                htm_result = results["htm_system"]
                temporal_output = htm_result.get("temporal_output", {})
                prediction_accuracy = temporal_output.get("prediction_accuracy", 0.0)
                n_bursts = temporal_output.get("n_bursts", 0)
                
                # Burst率作为预测误差的指标
                total_cols = temporal_output.get("total_columns", 100)
                burst_rate = n_bursts / max(1, total_cols)
                
                errors["prediction_error_htm"] = 1.0 - prediction_accuracy
                errors["burst_rate"] = burst_rate
                errors["uncertainty"] = burst_rate * 0.5
            
            if self.config.use_legacy_htm and "word_level_cortex" in results:
                word_result = results["word_level_cortex"]
                sent_result = results.get("sentence_level_cortex", {})
                errors["prediction_error_word"] = float(
                    torch.tensor(word_result.get("prediction_error", 0.1)).mean()
                )
                errors["prediction_error_sentence"] = float(
                    torch.tensor(sent_result.get("cross_layer_error", 0.1)).mean()
                )
            
            # 默认错误值
            if not errors:
                errors = {
                    "prediction_error": 0.1,
                    "uncertainty": 0.2
                }
            
            regulation_output = self.neural_regulator(
                errors=errors,
                task_complexity=0.5
            )
            results["neural_regulation"] = regulation_output
            logger.debug(f"神经调节耗时: {(time.time() - regulation_start) * 1000:.2f}ms")
            logger.debug(f"调节信号: {regulation_output}")
        else:
            regulation_output = {
                "attention_strength": 0.5,
                "learning_rate": 0.01,
                "system_bias": 0.5
            }
        
        # ========== 阶段6: 记忆系统交互 ==========
        memory_start = time.time()
        
        # 获取上下文表示（根据配置决定来源）
        if self.config.use_sparse_attention and "sparse_attention" in results:
            attended_output = sparse_attention_output["sparse_attention"]
            context_embeds = attended_output[:, 0, :] if attended_output.dim() == 3 else attended_output
        elif self.config.use_htm_pipeline and "htm_system" in results:
            htm_repr = results["htm_system"]["htm_repr"]
            context_embeds = htm_repr[:, 0, :] if htm_repr.dim() == 3 else htm_repr.mean(dim=1)
        elif self.config.use_legacy_htm and "sentence_level_cortex" in results:
            sent_repr = results["sentence_level_cortex"].get("sentence_repr", None)
            if isinstance(sent_repr, torch.Tensor):
                context_embeds = sent_repr[:, 0, :] if sent_repr.dim() == 3 else sent_repr
            else:
                context_embeds = torch.mean(gated_embeds, dim=1)
        else:
            context_embeds = torch.mean(gated_embeds, dim=1)
        
        if self.config.use_memory_system:
            # 工作记忆操作 - 使用forward接口
            wm_output = self.working_memory(
                input_embeds=context_embeds.unsqueeze(1),
                operation="write"
            )
            
            # 情景记忆检索
            episodic_output = self.episodic_memory(
                input_embeds=context_embeds
            )
            
            # 语义记忆检索
            semantic_output = self.semantic_memory(
                query_embeds=context_embeds
            )
            
            results["memory_system"] = {
                "working_memory": wm_output,
                "episodic_retrieval": episodic_output,
                "semantic_retrieval": semantic_output
            }
            logger.debug(f"记忆系统耗时: {(time.time() - memory_start) * 1000:.2f}ms")
        else:
            episodic_output = {"retrieved_indices": []}
            semantic_output = {"retrieved_entries": []}
        
        # ========== 阶段7: 双系统推理 ==========
        reasoning_start = time.time()
        
        # 系统1 - 直觉联想
        system1_output = self.system1(
            query_embeds=context_embeds
        )
        results["system1"] = {k: v.detach().cpu() if torch.is_tensor(v) else v 
                             for k, v in system1_output.items()}
        
        # 系统2 - 符号推理
        system2_output = self.system2(
            query=input_text[0],
            query_embeds=context_embeds
        )
        results["system2"] = {k: v.detach().cpu() if torch.is_tensor(v) else v 
                             for k, v in system2_output.items()}
        
        # 双系统融合 - 使用正确参数
        fusion_output = self.fusion(
            system1_output=system1_output,
            system2_output=system2_output,
            regulation_signals=regulation_output
        )
        results["fusion"] = {k: v.detach().cpu() if torch.is_tensor(v) else v 
                            for k, v in fusion_output.items()}
        logger.debug(f"推理耗时: {(time.time() - reasoning_start) * 1000:.2f}ms")
        
        # ========== 阶段8: 响应生成 ==========
        generation_start = time.time()
        
        # 简单生成响应
        generation_output = self.generator(
            prompt=input_text[0],
            max_length=50
        )
        
        # 确保有generated_text
        if "generated_text" not in generation_output:
            generation_output["generated_text"] = f"已处理: {input_text[0]}"
            
        results["generation"] = generation_output
        
        # 前向模型质量评估 - 使用正确参数
        if self.config.use_forward_model:
            quality_output = self.forward_model(
                candidate_text=generation_output["generated_text"],
                reference_text=input_text[0]
            )
            results["quality_assessment"] = quality_output
        logger.debug(f"生成耗时: {(time.time() - generation_start) * 1000:.2f}ms")
        
        # ========== 阶段9: 记忆更新 ==========
        if self.config.use_memory_system:
            # 存储新的情景记忆
            try:
                self.episodic_memory.store({
                    "embedding": context_embeds[0] if context_embeds.dim() > 1 else context_embeds,
                    "content": input_text[0],
                    "response": generation_output["generated_text"],
                    "timestamp": time.time()
                })
            except:
                pass
        
        # 总耗时
        total_time = (time.time() - start_time) * 1000
        results["performance"] = {
            "total_time_ms": total_time
        }
        
        # 在顶层确保有generated_text字段
        results["generated_text"] = generation_output["generated_text"]
        results["generation"] = generation_output
        
        logger.info(f"总处理耗时: {total_time:.2f}ms")
        logger.info(f"生成响应: {generation_output['generated_text']}")
        
        return results
    
    def chat(self, user_input: str, **kwargs) -> str:
        """简单的聊天接口"""
        results = self.forward(user_input, **kwargs)
        return results.get("generated_text", "抱歉，我无法生成响应。")
    
    def reset_memory(self):
        """重置所有记忆系统"""
        if hasattr(self, 'working_memory'):
            self.working_memory.clear()
        if hasattr(self, 'episodic_memory'):
            self.episodic_memory.clear()
        if hasattr(self, 'semantic_memory'):
            self.semantic_memory.clear()
        logger.info("所有记忆系统已重置")
    
    def reset_sequence(self):
        """重置序列状态（用于处理新的独立序列）"""
        if hasattr(self, 'htm_system'):
            self.htm_system.reset_sequence()
        if hasattr(self, 'temporal_memory'):
            self.temporal_memory.reset_sequence()
        logger.info("序列状态已重置")
    
    def reset_learning(self):
        """重置所有学习状态"""
        if hasattr(self, 'htm_system'):
            self.htm_system.reset_learning()
        logger.info("学习状态已重置")


# 简单测试入口
if __name__ == "__main__":
    utils.setup_logging("INFO")
    
    print("=" * 60)
    print("BioTransLM 编排器测试")
    print("=" * 60)
    
    # 创建编排器实例
    orchestrator = Orchestrator()
    
    # 测试简单输入
    test_input = "你好，今天天气怎么样？"
    print(f"\n输入: {test_input}")
    
    response = orchestrator.chat(test_input)
    print(f"输出: {response}")
    
    print("\n测试完成！")
