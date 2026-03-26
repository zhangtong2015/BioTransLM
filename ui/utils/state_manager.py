#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
状态管理器 - 管理 Gradio 会话状态
包含对话历史、训练状态、处理结果等
"""

import threading
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """对话消息"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_tuple(self) -> tuple:
        """转换为 Gradio Chatbot 所需的元组格式"""
        return (self.content, None) if self.role == 'user' else (None, self.content)


@dataclass
class Conversation:
    """一次完整的对话会话"""
    id: str
    messages: List[ChatMessage] = field(default_factory=list)
    processing_results: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """添加消息"""
        msg = ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(msg)
    
    def get_last_user_message(self) -> Optional[ChatMessage]:
        """获取最后一条用户消息"""
        for msg in reversed(self.messages):
            if msg.role == 'user':
                return msg
        return None
    
    def to_gradio_history(self) -> List[List[str]]:
        """转换为 Gradio Chatbot 兼容的格式"""
        history = []
        for i in range(0, len(self.messages), 2):
            user_msg = self.messages[i] if i < len(self.messages) and self.messages[i].role == 'user' else None
            ai_msg = self.messages[i+1] if i+1 < len(self.messages) and self.messages[i+1].role == 'assistant' else None
            
            if user_msg and ai_msg:
                history.append([user_msg.content, ai_msg.content])
            elif user_msg:
                history.append([user_msg.content, ""])
        return history


class StateManager:
    """全局状态管理器（线程安全）"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._state_lock = threading.RLock()
        
        # 对话状态
        self._conversations: Dict[str, Conversation] = {}
        self._current_conversation_id: Optional[str] = None
        
        # 训练状态
        self._training_state = {
            'is_training': False,
            'current_epoch': 0,
            'total_epochs': 0,
            'global_step': 0,
            'loss_history': [],
            'htm_metrics_history': [],
            'last_update': None
        }
        
        # 数据集状态
        self._dataset_state = {
            'loaded_dataset': None,
            'train_dataset': None,
            'eval_dataset': None,
            'preview_data': None,
            'statistics': None
        }
        
        self._initialized = True
        logger.info("StateManager 初始化完成")
    
    # ========== 对话管理 ==========
    
    def create_conversation(self) -> str:
        """创建新对话"""
        with self._state_lock:
            conv_id = str(uuid.uuid4())[:8]
            self._conversations[conv_id] = Conversation(id=conv_id)
            self._current_conversation_id = conv_id
            logger.info(f"创建新对话：{conv_id}")
            return conv_id
    
    def get_current_conversation(self) -> Optional[Conversation]:
        """获取当前对话"""
        with self._state_lock:
            if self._current_conversation_id:
                return self._conversations.get(self._current_conversation_id)
            return None
    
    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """获取指定对话"""
        with self._state_lock:
            return self._conversations.get(conv_id)
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """列出所有对话摘要"""
        with self._state_lock:
            return [
                {
                    'id': conv.id,
                    'message_count': len(conv.messages),
                    'created_at': conv.created_at,
                    'last_message_time': conv.messages[-1].timestamp if conv.messages else 0
                }
                for conv in self._conversations.values()
            ]
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """添加消息到当前对话"""
        with self._state_lock:
            conv = self.get_current_conversation()
            if not conv:
                conv_id = self.create_conversation()
                conv = self._conversations[conv_id]
            
            conv.add_message(role, content, metadata)
            return True
    
    def set_processing_result(self, result: Dict[str, Any]):
        """设置最近一次对话的处理结果"""
        with self._state_lock:
            conv = self.get_current_conversation()
            if conv:
                conv.processing_results = result
                logger.debug(f"已保存处理结果，包含 {len(result)} 个字段")
    
    def get_last_processing_result(self) -> Optional[Dict[str, Any]]:
        """获取最近一次对话的处理结果"""
        with self._state_lock:
            conv = self.get_current_conversation()
            if conv:
                return conv.processing_results
            return None
    
    def clear_conversations(self):
        """清空所有对话"""
        with self._state_lock:
            self._conversations.clear()
            self._current_conversation_id = None
            logger.info("已清空所有对话")
    
    # ========== 训练状态管理 ==========
    
    def start_training(self, total_epochs: int):
        """标记训练开始"""
        with self._state_lock:
            self._training_state['is_training'] = True
            self._training_state['total_epochs'] = total_epochs
            self._training_state['current_epoch'] = 0
            self._training_state['global_step'] = 0
            self._training_state['loss_history'] = []
            self._training_state['htm_metrics_history'] = []
            self._training_state['last_update'] = time.time()
            logger.info(f"训练开始，总 epoch 数：{total_epochs}")
    
    def update_training_state(self, 
                             epoch: int, 
                             step: int, 
                             losses: Dict[str, float],
                             htm_metrics: Optional[Dict[str, Any]] = None):
        """更新训练状态"""
        with self._state_lock:
            self._training_state['current_epoch'] = epoch
            self._training_state['global_step'] = step
            self._training_state['last_update'] = time.time()
            
            # 记录损失历史
            loss_record = {
                'step': step,
                'epoch': epoch,
                'timestamp': time.time(),
                **losses
            }
            self._training_state['loss_history'].append(loss_record)
            
            # 记录 HTM 指标
            if htm_metrics:
                htm_record = {
                    'step': step,
                    'timestamp': time.time(),
                    **htm_metrics
                }
                self._training_state['htm_metrics_history'].append(htm_record)
    
    def stop_training(self):
        """标记训练结束"""
        with self._state_lock:
            self._training_state['is_training'] = False
            self._training_state['last_update'] = time.time()
            logger.info("训练结束")
    
    def get_training_state(self) -> Dict[str, Any]:
        """获取训练状态"""
        with self._state_lock:
            return self._training_state.copy()
    
    def is_training(self) -> bool:
        """检查是否正在训练"""
        with self._state_lock:
            return self._training_state['is_training']
    
    # ========== 数据集状态管理 ==========
    
    def set_loaded_dataset(self, dataset: Any, preview: List[Dict], statistics: Dict[str, Any]):
        """设置已加载的数据集"""
        with self._state_lock:
            self._dataset_state['loaded_dataset'] = dataset
            self._dataset_state['preview_data'] = preview
            self._dataset_state['statistics'] = statistics
            logger.info(f"数据集已加载，{len(preview)} 条预览数据")
    
    def set_split_datasets(self, train_dataset: Any, eval_dataset: Any):
        """设置划分后的数据集"""
        with self._state_lock:
            self._dataset_state['train_dataset'] = train_dataset
            self._dataset_state['eval_dataset'] = eval_dataset
            logger.info("数据集已划分为训练集和评估集")
    
    def get_dataset_state(self) -> Dict[str, Any]:
        """获取数据集状态"""
        with self._state_lock:
            return self._dataset_state.copy()
    
    def clear_dataset(self):
        """清空数据集状态"""
        with self._state_lock:
            self._dataset_state = {
                'loaded_dataset': None,
                'train_dataset': None,
                'eval_dataset': None,
                'preview_data': None,
                'statistics': None
            }
            logger.info("数据集状态已清空")
    
    # ========== 通用方法 ==========
    
    def get_status_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        with self._state_lock:
            return {
                'conversations': len(self._conversations),
                'current_conversation': self._current_conversation_id,
                'is_training': self._training_state['is_training'],
                'training_progress': f"{self._training_state['current_epoch']}/{self._training_state['total_epochs']}" if self._training_state['total_epochs'] > 0 else "未开始",
                'has_dataset': self._dataset_state['loaded_dataset'] is not None
            }
