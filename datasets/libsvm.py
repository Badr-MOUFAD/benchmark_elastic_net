from benchopt import BaseDataset
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):

    name = "libsvm"

    parameters = {
        # ["bodyfat", "leukemia", "rcv1.binary", "news20.binary"],
        'dataset': ["rcv1.binary"],
    }

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata']

    def __init__(self, dataset="bodyfat"):
        self.dataset = dataset
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            self.X, self.y = fetch_libsvm(self.dataset)

        return dict(X=self.X, y=self.y)
