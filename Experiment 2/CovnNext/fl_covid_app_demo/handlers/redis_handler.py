import dill  # 0.3.2
import redis


class Redis:
    def __init__(self, host='127.0.0.1', port=6379, db=1, password='Comp@123'):


        if host is None:
            raise ValueError('RedisCache host parameter may not be None')
        if isinstance(host, str):
            self._client = redis.StrictRedis(host=host, port=port, db=db, password=password)
        else:
            self._client = host

    def has(self, key):
        return self._client.exists(key)

    def set(self, key, object):
        dump = self.dump_object(object)
        try:
            self._client.set(key, dump)
            success = True
        except:
            print("problem in redis")
            success = False

        return success

    def get(self, key):
        return self.load_object(self._client.get(key))

    def dump_object(self, object):

        serialized_object = None
        try:
            serialized_object = dill.dumps(object)
        except:
            print('Error Serializing --dill')
        return serialized_object


    def load_object(self, object):

        unserialized_object = None
        try:
            unserialized_object = dill.loads(object)
        except:
            print('Error Un-Serializing --dill')
        return unserialized_object
    def remove(self,key):
        self._client.delete(key)


