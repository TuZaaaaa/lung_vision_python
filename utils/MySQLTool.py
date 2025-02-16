import mysql.connector
from mysql.connector import Error
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

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
                database=self.database,
                autocommit=True  # 确保 insert, update, delete 立即生效
            )
            self.cursor = self.connection.cursor(dictionary=True)
        except Error as e:
            raise ConnectionError(f"连接 MySQL 失败: {str(e)}")

    def check_connection(self):
        """检查连接是否存活，避免长时间未使用导致失效"""
        try:
            self.connection.ping(reconnect=True, attempts=3, delay=2)
        except Error as e:
            self.__post_init__()  # 重新初始化连接

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> Dict:
        """执行 SQL 语句"""
        self.check_connection()
        try:
            self.cursor.execute(query, params or ())
            results = [self.cursor.fetchall()]
            while self.cursor.nextset():
                results.append(self.cursor.fetchall())
            self.connection.commit()
            return {"success": True, "affected_rows": self.cursor.rowcount, "data": results}
        except Error as e:
            return {"success": False, "error": str(e)}

    def fetch_one(self, query: str, params: Optional[Tuple] = None) -> Dict:
        """查询单行数据"""
        self.check_connection()
        try:
            self.cursor.execute(query, params or ())
            result = self.cursor.fetchone()
            return {"success": True, "data": result} if result else {"success": False, "data": None}
        except Error as e:
            return {"success": False, "error": str(e)}

    def fetch_all(self, query: str, params: Optional[Tuple] = None) -> Dict:
        """查询多行数据"""
        self.check_connection()
        try:
            self.cursor.execute(query, params or ())
            result = self.cursor.fetchall()
            return {"success": True, "data": result}
        except Error as e:
            return {"success": False, "error": str(e)}

    def insert(self, query: str, params: Tuple) -> Dict:
        """插入数据，并返回 last_insert_id"""
        self.check_connection()
        result = self.execute_query(query, params)
        if result["success"]:
            result["last_insert_id"] = self.last_insert_id()
        return result

    def update(self, query: str, params: Tuple) -> Dict:
        """更新数据"""
        return self.execute_query(query, params)

    def delete(self, query: str, params: Tuple) -> Dict:
        """删除数据"""
        return self.execute_query(query, params)

    def count_records(self, query: str, params: Optional[Tuple] = None) -> Dict:
        """统计记录数"""
        self.check_connection()
        try:
            self.cursor.execute(query, params or ())
            count = self.cursor.fetchone()
            return {"success": True, "count": count["count"] if count else 0}
        except Error as e:
            return {"success": False, "error": str(e)}

    def last_insert_id(self) -> int:
        """获取最新插入的 ID"""
        self.check_connection()
        self.cursor.execute("SELECT LAST_INSERT_ID() AS id;")
        result = self.cursor.fetchone()
        return result["id"] if result else 0

    def close_connection(self):
        """关闭数据库连接"""
        try:
            self.cursor.close()
            self.connection.close()
            return {"success": True}
        except Error as e:
            return {"success": False, "error": str(e)}
