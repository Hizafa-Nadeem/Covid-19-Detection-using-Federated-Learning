# import pickle

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import dill
import numpy as np  # 1.20.1
import pandas as pd  # 1.2.4
import time
import json
from managers.model_manager_2 import FLManager2
import base64


class CollectClientModelApiView(APIView):
    def post(self, request, *args, **kwargs):

        request_data = json.loads(request.body.decode("UTF-8"))
        ascii_encoded_bytes = request_data['local_grads']
        client_updates_bytes = base64.b64decode(ascii_encoded_bytes)
        client_updates = dill.loads(client_updates_bytes)
        client_updates = client_updates['local_grads']
        fl_manager = FLManager2()
        fl_manager.collect_gradients(client_updates)
        return Response({"updated": True}, status=status.HTTP_200_OK)


class GetGlobalModelApiView(APIView):
    def get(self, request, *args, **kwargs):
        # request_data = json.loads(request.body.decode("UTF-8"))
        # print(request_data['client_id'])
        fl_manager = FLManager2()
        global_model = fl_manager.get_global_model()
        request_response = {'global_model': global_model}
        return Response(request_response, status=status.HTTP_200_OK)
