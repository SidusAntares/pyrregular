from pyrregular.io_utils import read_csv
from pyrregular.reader_interface import ReaderInterface
from pyrregular.accessor import IrregularAccessor
import numpy as np

def read_your_dataset(filenames):
    data = np.load(filenames["data"])
    labels = np.load(filenames["labels"])
    time_ids = np.load(filenames["time_ids"])  # ← 新增：真实 DOY，shape=(N, T)

    ts_ids, signal_ids, t_indices = np.indices(data.shape)
    ts_ids = ts_ids.ravel()
    signal_ids = signal_ids.ravel()
    t_indices = t_indices.ravel()  # 这是时间维度的索引（0~T-1），不是时间戳！

    for ts_id, signal_id, t_idx in zip(ts_ids, signal_ids, t_indices):
        value = data[ts_id, signal_id, t_idx]
        if np.isnan(value):
            continue
        label = labels[ts_id]
        real_timestamp = time_ids[ts_id, t_idx]  # ← 关键：用真实 DOY！
        yield dict(
            time_series_id=ts_id,
            channel_id=signal_id,
            timestamp=int(real_timestamp),  # 确保是整数（DOY）
            value=value,
            labels=label,
        )

class convert_Dataset(ReaderInterface):
    @staticmethod
    def read_original_version(verbose=False):
        return read_csv(
            filenames={
                "data": "dataset_npy/dataset.npy",
                "labels": "dataset_npy/dataset_labels.npy",
                "time_ids":"dataset_npy/dataset_doy.npy",
            },
            ts_id="time_series_id",
            time_id="timestamp",
            signal_id="channel_id",
            value_id="value",
            dims={
                "ts_id": [
                    "labels"
                ],  # static variable that depends on the time series id
                "signal_id": [],
                "time_id": [],
            },
            reader_fun=read_your_dataset,
            time_index_as_datetime=False,
            verbose=verbose,
        )