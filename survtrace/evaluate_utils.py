from sksurv.metrics import concordance_index_ipcw, brier_score
import numpy as np
import pdb


class Evaluator:
    def __init__(self, df, train_index):
        '''the input duration_train should be the raw durations (continuous),
        NOT the discrete index of duration.
        '''
        self.df_train_all = df.loc[train_index]

    def eval_single(self, model, test_set, val_batch_size=None):
        df_train_all = self.df_train_all
        get_target = lambda df: (df['duration'].values, df['event'].values)
        durations_train, events_train = get_target(df_train_all)
        et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                        dtype = [('e', bool), ('t', float)])
        times = model.config['duration_index'][1:-1]
        horizons = model.config['horizons']

        df_test, df_y_test = test_set
        surv = model.predict_surv(df_test, batch_size=val_batch_size)
        risk = 1 - surv
        
        durations_test, events_test = get_target(df_y_test)
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

    def eval_multi(self, model, test_set, val_batch_size=10000):
        times = model.config['duration_index'][1:-1]
        horizons = model.config['horizons']
        df_train_all = self.df_train_all
        get_target = lambda df, risk: (df['duration'].values, df['event_{}'.format(risk)].values)
        df_test, df_y_test = test_set

        for risk_idx in range(model.config.num_event):
            durations_train, events_train = get_target(df_train_all, risk_idx)
            durations_test, events_test = get_target(df_y_test, risk_idx)
            
            surv = model.predict_surv(df_test, batch_size=val_batch_size, event=risk_idx)
            risk = 1 - surv

            et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                            dtype = [('e', bool), ('t', float)])
            et_test = np.array([(events_test[i], durations_test[i]) for i in range(len(events_test))],
                        dtype = [('e', bool), ('t', float)])
            cis = []
            for i, _ in enumerate(times):
                cis.append(concordance_index_ipcw(et_train, et_test, risk[:, i+1].to("cpu").numpy(), times[i])[0])

            brs = brier_score(et_train, et_test, surv.to("cpu").numpy()[:,1:-1], times)[1]

            for horizon in enumerate(horizons):
                print("Event: {} For {} quantile,".format(risk_idx,horizon[1]))
                print("TD Concordance Index - IPCW:", cis[horizon[0]])
                print("Brier Score:", brs[horizon[0]])

    def eval(self, model, test_set, val_batch_size=None):
        print("***"*10)
        print("start evaluation")
        print("***"*10)

        if model.config['num_event'] > 1:
            self.eval_multi(model, test_set, val_batch_size)
        else:
            self.eval_single(model, test_set, val_batch_size)

