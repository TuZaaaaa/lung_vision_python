import mysql.connector
from mysql.connector import Error
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

@dataclass
class MySQLTool:
    host: str = "localhost"
    user: str = "root"
    password: str = ""
    database: str = "test_db"
    connection: Any = field(init=False, repr=False)
    cursor: Any = field(init=False, repr=False)

    def __post_init__(self):
        """初始化 MySQL 连接"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.connection.cursor(dictionary=True)
        except Error as e:
            raise ConnectionError(f"连接 MySQL 失败: {str(e)}")

    def execute_query(self, query: str, params: Optional[tuple] = None) -> Dict:
        """执行 SQL 语句"""
        try:
            self.cursor.execute(query, params or ())
            self.connection.commit()
            return {"success": True, "affected_rows": self.cursor.rowcount}
        except Error as e:
            return {"success": False, "error": str(e)}

    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Dict:
        """查询单行数据"""
        try:
            self.cursor.execute(query, params or ())
            result = self.cursor.fetchone()
            return {"success": True, "data": result} if result else {"success": False, "data": None}
        except Error as e:
            return {"success": False, "error": str(e)}

    def fetch_all(self, query: str, params: Optional[tuple] = None) -> Dict:
        """查询多行数据"""
        try:
            self.cursor.execute(query, params or ())
            result = self.cursor.fetchall()
            return {"success": True, "data": result}
        except Error as e:
            return {"success": False, "error": str(e)}

    def insert(self, query: str, params: tuple) -> Dict:
        """插入数据"""
        return self.execute_query(query, params)

    def update(self, query: str, params: tuple) -> Dict:
        """更新数据"""
        return self.execute_query(query, params)

    def delete(self, query: str, params: tuple) -> Dict:
        """删除数据"""
        return self.execute_query(query, params)

    def count_records(self, query: str, params: Optional[tuple] = None) -> Dict:
        """统计记录数"""
        try:
            self.cursor.execute(query, params or ())
            count = self.cursor.fetchone()
            return {"success": True, "count": count["count"] if count else 0}
        except Error as e:
            return {"success": False, "error": str(e)}

    def close_connection(self):
        """关闭数据库连接"""
        try:
            self.cursor.close()
            self.connection.close()
            return {"success": True}
        except Error as e:
            return {"success": False, "error": str(e)}

