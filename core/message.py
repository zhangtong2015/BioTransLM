# Message基类 - 用于模块间通信
# 实现了任务表中要求的字段：id, timestamp, source, destination, type, payload, metadata, priority

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
from enum import Enum, auto
import time
import uuid
import json
from datetime import datetime
import torch


class MessageType(Enum):
    """消息类型枚举"""
    DATA = auto()
    CONTROL = auto()
    ERROR = auto()
    CONFIG = auto()
    REQUEST = auto()
    RESPONSE = auto()
    EVENT = auto()
    COMMAND = auto()


class Priority(Enum):
    """消息优先级枚举"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Message:
    """模块间通信的标准消息格式
    
    遵循任务表要求，包含以下字段：
    - id: 消息唯一标识符
    - timestamp: 时间戳
    - source: 源模块名称
    - destination: 目标模块名称
    - type: 消息类型
    - payload: 消息数据负载
    - metadata: 额外元数据
    - priority: 消息优先级
    """
    
    source: str
    destination: str
    payload: Union[Dict[str, Any], torch.Tensor, str, bytes]
    type: MessageType = MessageType.DATA
    priority: Priority = Priority.NORMAL
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后验证"""
        if isinstance(self.payload, torch.Tensor):
            self.metadata["is_tensor"] = True
            self.metadata["tensor_shape"] = tuple(self.payload.shape)
            self.metadata["tensor_dtype"] = str(self.payload.dtype)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "id": self.id,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "source": self.source,
            "destination": self.destination,
            "type": self.type.name if isinstance(self.type, MessageType) else self.type,
            "priority": self.priority.value if isinstance(self.priority, Priority) else self.priority,
            "metadata": self.metadata
        }
        
        # 处理payload（张量特殊处理）
        if isinstance(self.payload, torch.Tensor):
            result["payload_type"] = "tensor"
            result["payload"] = f"Tensor(shape={self.payload.shape}, dtype={self.payload.dtype})"
        elif isinstance(self.payload, bytes):
            result["payload_type"] = "bytes"
            result["payload"] = f"Bytes(length={len(self.payload)})"
        else:
            result["payload_type"] = type(self.payload).__name__
            result["payload"] = self.payload
        
        return result
    
    def to_json(self) -> str:
        """转换为JSON字符串（用于序列化）"""
        data = self.to_dict()
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def is_tensor_payload(self) -> bool:
        """检查payload是否为张量"""
        return isinstance(self.payload, torch.Tensor)
    
    def get_tensor_payload(self) -> Optional[torch.Tensor]:
        """获取张量payload"""
        if self.is_tensor_payload():
            return self.payload
        return None
    
    def get_dict_payload(self) -> Dict[str, Any]:
        """获取字典格式的payload（安全访问）"""
        if isinstance(self.payload, dict):
            return self.payload
        return {"value": self.payload}
    
    def age(self) -> float:
        """获取消息年龄（秒）"""
        return time.time() - self.timestamp
    
    def age_ms(self) -> float:
        """获取消息年龄（毫秒）"""
        return self.age() * 1000
    
    def __str__(self) -> str:
        """字符串表示"""
        payload_preview = ""
        if isinstance(self.payload, torch.Tensor):
            payload_preview = f"Tensor(shape={self.payload.shape})"
        elif isinstance(self.payload, dict):
            keys = list(self.payload.keys())
            payload_preview = f"Dict(keys={keys[:5]}{'...' if len(keys) > 5 else ''})"
        elif isinstance(self.payload, str):
            payload_preview = self.payload[:50] + "..." if len(self.payload) > 50 else self.payload
        else:
            payload_preview = str(type(self.payload).__name__)
        
        return (
            f"Message(id={self.id[:8]}..., "
            f"type={self.type.name if isinstance(self.type, MessageType) else self.type}, "
            f"{self.source} -> {self.destination}, "
            f"priority={self.priority.name if isinstance(self.priority, Priority) else self.priority}, "
            f"payload={payload_preview}, "
            f"age={self.age_ms():.2f}ms)"
        )
    
    def __repr__(self) -> str:
        return self.__str__()


class MessageBus:
    """消息总线 - 用于模块间发布订阅通信"""
    
    def __init__(self, max_history: int = 1000):
        self._subscribers: Dict[str, list] = {}  # topic -> [callbacks]
        self._message_history: list[Message] = []
        self._max_history = max_history
        self._global_subscribers: list = []
    
    def subscribe(self, topic: str, callback):
        """订阅特定主题"""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)
    
    def subscribe_all(self, callback):
        """订阅所有主题"""
        self._global_subscribers.append(callback)
    
    def unsubscribe(self, topic: str, callback):
        """取消订阅"""
        if topic in self._subscribers and callback in self._subscribers[topic]:
            self._subscribers[topic].remove(callback)
    
    def publish(self, topic: str, message: Message):
        """发布消息到特定主题"""
        # 记录历史
        self._message_history.append(message)
        if len(self._message_history) > self._max_history:
            self._message_history.pop(0)
        
        # 通知主题订阅者
        if topic in self._subscribers:
            for callback in self._subscribers[topic]:
                try:
                    callback(message)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"消息回调执行失败: {e}", exc_info=True)
        
        # 通知全局订阅者
        for callback in self._global_subscribers:
            try:
                callback(topic, message)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"全局消息回调执行失败: {e}", exc_info=True)
    
    def get_history(self, topic: Optional[str] = None, 
                    source: Optional[str] = None,
                    msg_type: Optional[MessageType] = None) -> list[Message]:
        """获取消息历史"""
        history = self._message_history
        
        if topic:
            # 注意：这里简化处理，实际应该记录每个消息的topic
            pass
        
        if source:
            history = [m for m in history if m.source == source]
        
        if msg_type:
            history = [m for m in history if m.type == msg_type]
        
        return history
    
    def clear_history(self):
        """清除历史"""
        self._message_history.clear()


# 便捷函数
def create_data_message(source: str, destination: str, 
                        payload: Dict[str, Any], **kwargs) -> Message:
    """创建数据消息"""
    return Message(
        source=source,
        destination=destination,
        payload=payload,
        type=MessageType.DATA,
        **kwargs
    )


def create_control_message(source: str, destination: str, 
                           command: str, params: Dict[str, Any] = None, **kwargs) -> Message:
    """创建控制消息"""
    payload = {
        "command": command,
        "params": params or {}
    }
    return Message(
        source=source,
        destination=destination,
        payload=payload,
        type=MessageType.CONTROL,
        **kwargs
    )


def create_error_message(source: str, destination: str, 
                         error: str, details: Dict[str, Any] = None, **kwargs) -> Message:
    """创建错误消息"""
    payload = {
        "error": error,
        "details": details or {}
    }
    return Message(
        source=source,
        destination=destination,
        payload=payload,
        type=MessageType.ERROR,
        priority=Priority.HIGH,
        **kwargs
    )


# 简单测试
if __name__ == "__main__":
    # 创建消息
    msg = create_data_message(
        source="test_module",
        destination="receiver",
        payload={"key": "value", "data": [1, 2, 3]}
    )
    print(msg)
    print("\n消息字典:")
    print(json.dumps(msg.to_dict(), indent=2, ensure_ascii=False))
    
    # 带张量的消息
    import torch
    tensor_msg = Message(
        source="encoder",
        destination="decoder",
        payload=torch.randn(2, 768),
        type=MessageType.DATA,
        priority=Priority.HIGH
    )
    print("\n带张量的消息:")
    print(tensor_msg)
    print(f"是张量payload: {tensor_msg.is_tensor_payload()}")
