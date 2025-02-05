from dataclasses import dataclass, asdict
from typing import Generic, TypeVar

from flask import jsonify

T = TypeVar('T')


@dataclass
class Result(Generic[T]):
    success: bool
    msg: str = None
    data: T = None

    @staticmethod
    def success() -> 'Result[T]':
        return Result(success=True)

    @staticmethod
    def success_with_data(data: T) -> 'Result[T]':
        return Result(success=True, data=data)

    @staticmethod
    def error(msg: str) -> 'Result[T]':
        return Result(success=False, msg=msg)

    def to_response(self, status_code: int = 200):
        """
        将 Result 对象转换为 Flask 的 Response 对象。
        :param status_code: HTTP 状态码，默认为 200
        :return: Flask Response 对象
        """
        return jsonify(asdict(self)), status_code
