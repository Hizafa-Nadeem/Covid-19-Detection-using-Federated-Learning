from django.conf import settings
import numpy as np
from handlers.redis_handler import Redis
from FL_models.models import Sum

redis = Redis(host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB, password=settings.REDIS_PASSWORD)


class FLManager:
    def __init__(self):
        self.model_updates_counter = 3
        self.model_updates = []
        self.model = Sum()
        self.model_updates_key = 'model_updates'
        self.counter_key = 'model_updates_counter'
        self.model_key = 'model'
        if not (redis.has(self.model_key) and redis.has(self.model_updates_key) and redis.has(self.counter_key)):
            redis.set(self.model_key, self.model)
            redis.set(self.model_updates_key, self.model_updates)
            redis.set(self.counter_key, self.model_updates_counter)

    def update_global_model(self):
        isUpdated = False
        print("Update global model")
        if redis.has(self.model_key) and redis.has(self.model_updates_key) and redis.has(self.counter_key):

            self.model = redis.get(self.model_key)
            self.model_updates = redis.get(self.model_updates_key)
            self.model_updates_counter = redis.get(self.counter_key)
            if len(self.model_updates) >= self.model_updates_counter:
                print(np.sum(self.model_updates))
                self.model.update_sum(np.sum(self.model_updates))
                self.model_updates = []
                redis.set(self.model_updates_key, self.model_updates)
                redis.set(self.model_key, self.model)
                isUpdated = True
        return isUpdated

    def get_global_model(self):
        print("Fetch global model")
        if self.update_global_model():
            if redis.has(self.model_key):
                print("read global model from redis")
                self.model = redis.get(self.model_key)
                print(self.model.get_sum())
        return self.model.get_sum()

    def collect_client_model_updates(self,client_updates=None):
        isCollected = False
        if redis.has(self.model_updates_key):
            self.model_updates = redis.get(self.model_updates_key)
            print(client_updates)
            self.model_updates.append(client_updates)
            redis.set(self.model_updates_key, self.model_updates)
            isCollected = True
        print("Collect client model updates")
        return isCollected

    def collect_client_performance_metrics(self):
        print("Collect client performance metrics")