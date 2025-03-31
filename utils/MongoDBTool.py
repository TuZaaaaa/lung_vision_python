from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

@dataclass
class MongoDBTool:
    uri: str = "mongodb://localhost:27017"
    db_name: str = "test_db"
    collection_name: str = "test_collection"
    client: MongoClient = field(init=False, repr=False)
    db: Any = field(init=False, repr=False)
    collection: Any = field(init=False, repr=False)

    def __post_init__(self):
        """初始化 MongoDB 连接"""
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
        except PyMongoError as e:
            raise ConnectionError(f"连接 MongoDB 失败: {str(e)}")

    def insert_one(self, document: Dict) -> Dict:
        """插入单个文档"""
        try:
            result = self.collection.insert_one(document)
            return {"success": True, "inserted_id": str(result.inserted_id)}
        except PyMongoError as e:
            return {"success": False, "error": str(e)}

    def insert_many(self, documents: List[Dict]) -> Dict:
        """批量插入文档"""
        try:
            result = self.collection.insert_many(documents)
            return {"success": True, "inserted_ids": [str(id) for id in result.inserted_ids]}
        except PyMongoError as e:
            return {"success": False, "error": str(e)}

    def find_one(self, filter_query: Optional[Dict] = None, projection: Optional[Dict] = None) -> Dict:
        """查询单个文档"""
        try:
            result = self.collection.find_one(filter_query or {}, projection)
            return {"success": True, "data": result} if result else {"success": False, "data": None}
        except PyMongoError as e:
            return {"success": False, "error": str(e)}

    def find_all(self, filter_query: Optional[Dict] = None, projection: Optional[Dict] = None,
                 sort: Optional[List[tuple]] = None, limit: int = 0) -> Dict:
        """查询所有匹配文档"""
        try:
            cursor = self.collection.find(filter_query or {}, projection)
            if sort:
                cursor = cursor.sort(sort)
            if limit > 0:
                cursor = cursor.limit(limit)
            return {"success": True, "data": list(cursor)}
        except PyMongoError as e:
            return {"success": False, "error": str(e)}

    def update_one(self, filter_query: Dict, update_data: Dict, upsert: bool = False) -> Dict:
        """更新单个文档"""
        try:
            result = self.collection.update_one(filter_query, {"$set": update_data}, upsert=upsert)
            return {"success": True, "matched_count": result.matched_count, "modified_count": result.modified_count}
        except PyMongoError as e:
            return {"success": False, "error": str(e)}

    def delete_one(self, filter_query: Dict) -> Dict:
        """删除单个文档"""
        try:
            result = self.collection.delete_one(filter_query)
            return {"success": True, "deleted_count": result.deleted_count}
        except PyMongoError as e:
            return {"success": False, "error": str(e)}

    def delete_many(self, filter_query: Dict) -> Dict:
        """批量删除文档"""
        try:
            result = self.collection.delete_many(filter_query)

            return {"success": True, "deleted_count": result.deleted_count}
        except PyMongoError as e:
            return {"success": False, "error": str(e)}

    def paginated_query(self, filter_query: Optional[Dict] = None, page: int = 1, per_page: int = 10,
                        sort: Optional[List[tuple]] = None) -> Dict:
        """分页查询"""
        try:
            total = self.collection.count_documents(filter_query or {})
            skip = (page - 1) * per_page
            cursor = self.collection.find(filter_query or {})
            if sort:
                cursor = cursor.sort(sort)
            cursor = cursor.skip(skip).limit(per_page)
            return {"success": True, "data": list(cursor), "total": total, "page": page, "per_page": per_page}
        except PyMongoError as e:
            return {"success": False, "error": str(e)}

    def create_index(self, keys: List[tuple], **kwargs) -> Dict:
        """创建索引"""
        try:
            index_name = self.collection.create_index(keys, **kwargs)
            return {"success": True, "index_name": index_name}
        except PyMongoError as e:
            return {"success": False, "error": str(e)}

    def count_documents(self, filter_query: Optional[Dict] = None) -> Dict:
        """统计文档数量"""
        try:
            count = self.collection.count_documents(filter_query or {})
            return {"success": True, "count": count}
        except PyMongoError as e:
            return {"success": False, "error": str(e)}

    def close_connection(self):
        """关闭数据库连接"""
        try:
            self.client.close()
            return {"success": True}
        except PyMongoError as e:
            return {"success": False, "error": str(e)}
