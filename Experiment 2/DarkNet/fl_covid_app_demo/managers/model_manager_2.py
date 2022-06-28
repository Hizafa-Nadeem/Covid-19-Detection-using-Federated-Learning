import dill
from django.conf import settings
import numpy as np
from handlers.redis_handler import Redis
from FL_models.model import FLServer
from os.path import exists

redis = Redis(host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB, password=settings.REDIS_PASSWORD)
#CONFIG SET requirepass "Comp@123"
#REDIS SET PASSWORD EVERYTIME YOU START REDIS SERVER ON WINDOWS


class FLManager2:
    def __init__(self):
        self.model_key = 'fl_server_attr'
        self.fl_server = FLServer()

        if exists("ae.pt"):
            self.fl_server.load_model()

        if redis.has(self.model_key):
            s_obj = redis.get(self.model_key)
            self.fl_server.load_object(s_obj)
            print("Optimizer Loaded From Redis")

    def get_global_model(self):

        key = "new_grad"
        if redis.has("new_grad"):
            s_new_grad = redis.get("new_grad")
            u_new_grad = dill.loads(s_new_grad)
            self.fl_server.update_grad(u_new_grad["new_grad"])
            self.fl_server.train_model()
            self.fl_server.save_model()
            redis.remove(key)
        return self.fl_server.get_weights()

    def collect_gradients(self, local_gradient): #only storing gradients

        if redis.has("new_grad"):
            s_new_grad = redis.get("new_grad")
            u_new_grad = dill.loads(s_new_grad)

            grad_arr = []
            grad_arr.append(u_new_grad["new_grad"])
            grad_arr.append(local_gradient)

            new_grad = [np.zeros_like(x) for x in local_gradient]

            for gradient in grad_arr:
                for i in range(len(gradient)):
                    new_grad[i] += np.array(gradient[i])

            try:
                s_obj = dill.dumps({
                    'new_grad': new_grad
                })
            except:
                print("Error saving gradient in redis")
        else:
            try:
                s_obj = dill.dumps({
                    'new_grad': local_gradient
                })
            except:
                print("Error saving gradient in redis")

        redis.set("new_grad", s_obj)
        if redis.set("new_grad", s_obj) is True:
            print("New_gradient Saved In Redis")

        #  self.fl_server.update_grad(new_grad)

        #self.fl_server.train_model()
        #self.fl_server.save_model()
        '''s_obj = self.fl_server.dump_object()'''





