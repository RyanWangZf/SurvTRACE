from sksurv.metrics import concordance_index_ipcw, brier_score
import numpy as np
import pdb


class Evaluator:
    def __init__(self, df, train_index):
        '''the input duration_train should be the raw durations (continuous),
        NOT the discrete index of duration.
        '''
        self.get_target = lambda df: (df['duration'].values, df['event'].values)
        durations_train, events_train = self.get_target(df.loc[train_index])

        # evaluate model by TD Concordance Index - IPCW
        self.et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                        dtype = [('e', bool), ('t', float)])

    def eval(self, model, test_set, val_batch_size=None):
        times = model.config['duration_index'][1:-1]
        horizons = model.config['horizons']
        et_train = self.et_train

        df_test, df_y_test = test_set
        surv = model.predict_surv(df_test, batch_size=val_batch_size)
        risk = 1 - surv
        
        durations_test, events_test = self.get_target(df_y_test)
        et_test = np.array([(events_test[i], durations_test[i]) for i in range(len(events_test))],
                    dtype = [('e', bool), ('t', float)])

        cis = []
        for i, _ in enumerate(times):
            cis.append(
                concordance_index_ipcw(et_train, et_test, estimate=risk[:, i+1].to("cpu").numpy(), tau=times[i])[0]
                )

        brs = brier_score(et_train, et_test, surv.to("cpu").numpy()[:,1:-1], times)[1]

        for horizon in enumerate(horizons):
            print(f"For {horizon[1]} quantile,")
            print("TD Concordance Index - IPCW:", cis[horizon[0]])
            print("Brier Score:", brs[horizon[0]])

