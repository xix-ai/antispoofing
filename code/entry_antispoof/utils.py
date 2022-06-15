import copy


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetaData(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.state_dict = None
        self.loss = float("inf")
        self.fbeta = float("-inf")
        self.min_c = float("inf")
        self.epoch = 0

    def update(self, state_dict, loss, fbeta, min_c, epoch):
        self.state_dict = copy.deepcopy(state_dict)
        self.loss = loss
        self.fbeta = fbeta
        self.min_c = min_c
        self.epoch = epoch

    def __str__(self):
        return "epoch_{}_loss_{:.4f}_fbeta_{:.4f}_min_c_{:.4f}".format(
            self.epoch, self.loss, self.fbeta, self.min_c
        )
