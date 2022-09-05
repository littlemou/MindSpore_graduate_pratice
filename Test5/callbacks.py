from mindspore.train.callback import Callback
from mindspore import save_checkpoint
import os, stat, copy

# 记录模型accuracy
class TrainHistroy(Callback):
    
    def __init__(self, history):
        super(TrainHistroy, self).__init__()
        self.history = history
        
    # 每个epoch结束时执行
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs.asnumpy()
        self.history.append(loss)
        
        
# 测试并记录模型在测试集的loss和accuracy，每个epoch结束时进行模型测试并记录结果，跟踪并保存准确率最高的模型的网络参数
class EvalHistory(Callback):
    #保存accuracy最高的网络参数
    best_param = None
    best_param_backbone = None
    
    def __init__(self, model, backbone, loss_history, acc_history, eval_data):
        super(EvalHistory, self).__init__()
        self.model = model
        self.backbone = backbone
        self.loss_history = loss_history
        self.acc_history = acc_history
        self.eval_data = eval_data
    
    # 每个epoch结束时执行
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        res = self.model.eval(self.eval_data, dataset_sink_mode=False)
        
        if len(self.acc_history)==0:
            self.best_param = copy.deepcopy(cb_params.network)
            self.best_param_backbone = copy.deepcopy(self.backbone)
        elif res['accuracy']>=max(self.acc_history):
            self.best_param = copy.deepcopy(cb_params.network)
            self.best_param_backbone = copy.deepcopy(self.backbone)

        self.loss_history.append(res['loss'])
        self.acc_history.append(res['accuracy'])
        
        print('acc_eval: ',res['accuracy'])
    
    # 训练结束后执行
    def end(self, run_context):
        # 保存最优网络参数
        if os.path.exists('best_param.ckpt'):
            os.chmod('best_param.ckpt', stat.S_IWRITE)
        if os.path.exists('best_param_backbone.ckpt'):
            os.chmod('best_param_backbone.ckpt', stat.S_IWRITE)
        save_checkpoint(self.best_param, 'best_param.ckpt')
        save_checkpoint(self.best_param_backbone, 'best_param_backbone.ckpt')