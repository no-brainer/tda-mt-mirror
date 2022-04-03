class MetricTracker:

    def __init__(self, keys):
        self.data = {key: None for key in keys}
        self.reset()

    def reset(self):
        for key in self.data.keys():
            self.data[key] = dict(total=0, count=0, avg=0)

    def update(self, key, value, n=1):
        self.data[key]["total"] += n * value
        self.data[key]["count"] += n
        self.data[key]["avg"] = self.data[key]["total"] / self.data[key]["count"]

    def all_avg(self):
        return {key: self.data[key]["avg"] for key in self.data.keys()}
