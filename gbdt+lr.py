import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')
import gc
from scipy import sparse

def preProcess():
    path = 'data/'
    print('读取数据...')
    df_train = pd.read_csv(path + 'xl_tag.csv')
    df_test = pd.read_csv(path + 'pf_tag.csv')
    print(type(df_train))
    #df_train.replace('-1', 0, inplace=True)
    #df_train.replace('-1', 0, inplace=True)
    df_train.replace('\\N',0,inplace=True)
    df_test.replace('\\N',0,inplace=True)
    #df_train[df_train["job_year"]=="\\N"]["job_year"] = "0"
    #df_train["job_year"] = df_train["job_year"].map({"\\N":"0"})
    # print(len(df_train[df_train["job_year"]=="\\N"]))
    #df_train[df_train["job_year"] == '99']["job_year"] = 45
    # df_train[df_train["hav_car_grp_ind"] == '\\N']["hav_car_grp_ind"] = 0
    # df_train[df_train["hav_hou_grp_ind"] == '\\N']["hav_hou_grp_ind"] = 0
    # df_test[df_test["job_year"] == '\\N']["job_year"] = 0
    # df_test[df_test["hav_car_grp_ind"] == '\\N']["hav_car_grp_ind"] = 0
    # df_test[df_test["hav_hou_grp_ind"] == '\\N']["hav_hou_grp_ind"] = 0
    print('读取结束')
    df_train.drop(['id'], axis = 1, inplace = True)
    df_test.drop(['id'], axis = 1, inplace = True)

    df_test['Label'] = -1

    data = pd.concat([df_train, df_test])
    data = data.fillna(-1)
    data.to_csv('data/data.csv', index = False)
    return data

def lr_predict(data, category_feature, continuous_feature): # 0.47181
    # 连续特征归一化
    print('开始归一化...')
    scaler = MinMaxScaler()
    for col in continuous_feature:
        print(col)
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    print('归一化结束')
    
    # 离散特征one-hot编码
    print('开始one-hot...')
    for col in category_feature:
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)
    print('one-hot结束')

    train = data[data['flag'] != -1]
    target = train.pop('flag')
    test = data[data['flag'] == -1]
    test.drop(['flag'], axis = 1, inplace = True)

    # 划分数据集
    print('划分数据集...')
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state = 2018)
    print('开始训练...')
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    print('tr-logloss: ', tr_logloss)
    val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print('val-logloss: ', val_logloss)
    print('开始预测...')
    y_pred = lr.predict_proba(test)[:, 1]
    print('写入结果...')
    #res = pd.read_csv('data/test.csv')
    #submission = pd.DataFrame({'id': test["id"], 'flag': y_pred})
    #print(test["id"])
    #print(submission)

    for i in y_pred:
        print(i)
    #print(y_pred)
    #print(type(res['id']))
    #print(type(y_pred))
    #submission.to_csv('submission/submission_lr_trlogloss_%s_vallogloss_%s.csv' % (tr_logloss, val_logloss), index = False)
    print('结束')

def gbdt_predict(data, category_feature, continuous_feature): # 0.44548
    # 离散特征one-hot编码
    print('开始one-hot...')
    for col in category_feature:
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)
    print('one-hot结束')

    train = data[data['flag'] != -1]
    target = train.pop('flag')
    test = data[data['flag'] == -1]
    test.drop(['flag'], axis = 1, inplace = True)

    # 划分数据集
    print('划分数据集...')
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state = 2018)

    print('开始训练..')
    gbm = lgb.LGBMClassifier(objective='binary',
                            subsample= 0.8,
                            min_child_weight= 0.5,
                            colsample_bytree= 0.7,
                            num_leaves=100,
                            max_depth = 12,
                            learning_rate=0.01,
                            n_estimators=10000,
                            )

    gbm.fit(x_train, y_train,
            eval_set = [(x_train, y_train), (x_val, y_val)],
            eval_names = ['train', 'val'],
            eval_metric = 'binary_logloss',
            early_stopping_rounds = 100,
            )
    tr_logloss = log_loss(y_train, gbm.predict_proba(x_train)[:, 1])
    val_logloss = log_loss(y_val, gbm.predict_proba(x_val)[:, 1])
    y_pred = gbm.predict_proba(test)[:, 1]
    print('写入结果...')
    res = pd.read_csv('data/test.csv')
    submission = pd.DataFrame({'id': res['id'], 'flag': y_pred})
    submission.to_csv('submission/submission_gbdt_trlogloss_%s_vallogloss_%s.csv' % (tr_logloss, val_logloss), index = False)
    print('结束')

def gbdt_lr_predict(data, category_feature, continuous_feature): # 0.43616
    # 离散特征one-hot编码
    print('开始one-hot...')
    for col in category_feature:
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)
    print('one-hot结束')

    train = data[data['flag'] != -1]
    target = train.pop('flag')
    test = data[data['flag'] == -1]
    test.drop(['flag'], axis = 1, inplace = True)

    # 划分数据集
    print('划分数据集...')
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state = 2018)

    print('开始训练gbdt..')
    gbm = lgb.LGBMRegressor(objective='binary',
                            subsample= 0.8,
                            min_child_weight= 0.5,
                            colsample_bytree= 0.7,
                            num_leaves=100,
                            max_depth = 12,
                            learning_rate=0.05,
                            n_estimators=10,
                            )

    gbm.fit(x_train, y_train,
            eval_set = [(x_train, y_train), (x_val, y_val)],
            eval_names = ['train', 'val'],
            eval_metric = 'binary_logloss',
            # early_stopping_rounds = 100,
            )
    model = gbm.booster_
    print('训练得到叶子数')
    gbdt_feats_train = model.predict(train, pred_leaf = True)
    gbdt_feats_test = model.predict(test, pred_leaf = True)
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns = gbdt_feats_name) 
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns = gbdt_feats_name)

    print('构造新的数据集...')
    train = pd.concat([train, df_train_gbdt_feats], axis = 1)
    test = pd.concat([test, df_test_gbdt_feats], axis = 1)
    train_len = train.shape[0]
    data = pd.concat([train, test])
    del train
    del test
    gc.collect()

    # # 连续特征归一化
    # print('开始归一化...')
    # scaler = MinMaxScaler()
    # for col in continuous_feature:
    #     data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    # print('归一化结束')

    # 叶子数one-hot
    print('开始one-hot...')
    for col in gbdt_feats_name:
        print('this is feature:', col)
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)
    print('one-hot结束')

    train = data[: train_len]
    test = data[train_len:]
    del data
    gc.collect()

    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.3, random_state = 2018)
    # lr
    print('开始训练lr..')
    lr = LogisticRegression()

    #lbl = preprocessing.LabelEncoder()
    #x_train['hav_car_grp_ind'] = lbl.fit_transform(x_train['hav_car_grp_ind'].astype(str))
    #x_train['hav_hou_grp_ind'] = lbl.fit_transform(x_train['hav_hou_grp_ind'].astype(str))
    #x_train['job_year'] = lbl.fit_transform(x_train['job_year'].astype(str))
    lr.fit(x_train, y_train)
    tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    print('tr-logloss: ', tr_logloss)
    val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print('val-logloss: ', val_logloss)
    print('开始预测...')
    y_pred = lr.predict_proba(test)[:, 1]
    print('写入结果...')
    for i in y_pred:
        print(i)
    #res = pd.read_csv('data/test.csv')
    #submission = pd.DataFrame({'id': res['id'], 'flag': y_pred})
    #submission.to_csv('submission/submission_gbdt+lr_trlogloss_%s_vallogloss_%s.csv' % (tr_logloss, val_logloss), index = False)
    print('结束')

def gbdt_ffm_predict(data, category_feature, continuous_feature):
    # 离散特征one-hot编码
    print('开始one-hot...')
    for col in category_feature:
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data = pd.concat([data, onehot_feats], axis = 1)
    print('one-hot结束')

    feats = [col for col in data if col not in category_feature] # onehot_feats + continuous_feature
    tmp = data[feats]
    train = tmp[tmp['flag'] != -1]
    target = train.pop('flag')
    test = tmp[tmp['flag'] == -1]
    test.drop(['flag'], axis = 1, inplace = True)
    
    # 划分数据集
    print('划分数据集...')
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state = 2018)

    print('开始训练gbdt..')
    gbm = lgb.LGBMRegressor(objective='binary',
                            subsample= 0.8,
                            min_child_weight= 0.5,
                            colsample_bytree= 0.7,
                            num_leaves=100,
                            max_depth = 12,
                            learning_rate=0.05,
                            n_estimators=10,
                            )

    gbm.fit(x_train, y_train,
            eval_set = [(x_train, y_train), (x_val, y_val)],
            eval_names = ['train', 'val'],
            eval_metric = 'binary_logloss',
            # early_stopping_rounds = 100,
            )
    model = gbm.booster_
    print('训练得到叶子数')
    gbdt_feats_train = model.predict(train, pred_leaf = True)
    gbdt_feats_test = model.predict(test, pred_leaf = True)
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns = gbdt_feats_name) 
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns = gbdt_feats_name)

    print('构造新的数据集...')
    tmp = data[category_feature + continuous_feature + ['flag']]
    train = tmp[tmp['flag'] != -1]
    test = tmp[tmp['flag'] == -1]
    train = pd.concat([train, df_train_gbdt_feats], axis = 1)
    test = pd.concat([test, df_test_gbdt_feats], axis = 1)
    data = pd.concat([train, test])
    del train
    del test
    gc.collect()

    # 连续特征归一化
    print('开始归一化...')
    scaler = MinMaxScaler()
    for col in continuous_feature:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    print('归一化结束')

    data.to_csv('data/data.csv', index = False)
    return category_feature + gbdt_feats_name

def FFMFormat(df, flag, path, train_len, category_feature = [], continuous_feature = []):
    index = df.shape[0]
    train = open(path + 'train.ffm', 'w')
    test = open(path + 'test.ffm', 'w')
    feature_index = 0
    feat_index = {}
    for i in range(index):
        feats = []
        field_index = 0
        for j, feat in enumerate(category_feature):
            t = feat + '_' + str(df[feat][i])
            if t not in  feat_index.keys():
                feat_index[t] = feature_index
                feature_index = feature_index + 1
            feats.append('%s:%s:%s' % (field_index, feat_index[t], 1))
            field_index = field_index + 1

        for j, feat in enumerate(continuous_feature):
            feats.append('%s:%s:%s' % (field_index, feature_index, df[feat][i]))
            feature_index = feature_index + 1
            field_index = field_index + 1

        print('%s %s' % (df[flag][i], ' '.join(feats)))

        if i < train_len:
            train.write('%s %s\n' % (df[flag][i], ' '.join(feats)))
        else:
            test.write('%s\n' % (' '.join(feats)))
    train.close()
    test.close()
   
    
if __name__ == '__main__':
    data = preProcess()
    #continuous_feature = ['I'] * 13
    #continuous_feature = [col + str(i + 1) for i, col in enumerate(continuous_feature)]
    continuous_feature = ["perm_crd_lmt_cd","age","l6mon_daim_aum_cd","perm_crd_lmt_cd","cur_debit_cnt",
                          "cur_credit_cnt","cur_debit_min_opn_dt_cnt","cur_credit_min_opn_dt_cnt","cur_debit_crd_lvl"]
    #category_feature = ['C'] * 26
    #category_feature = [col + str(i + 1) for i, col in enumerate(category_feature)]
    category_feature = ["gdr_cd","mrg_situ_cd","edu_deg_cd","acdm_deg_cd","deg_cd","fr_or_sh_ind","dnl_mbl_bnk_ind",
                        "dnl_bind_cmb_lif_ind","l6mon_agn_ind","vld_rsk_ases_ind","fin_rsk_ases_grd_cd",
                        "confirm_rsk_ases_lvl_typ_cd","cust_inv_rsk_endu_lvl_cd","tot_ast_lvl_cd","pot_ast_lvl_cd",
                        "bk1_cur_year_mon_avg_agn_amt_cd","l12mon_buy_fin_mng_whl_tms","l12_mon_fnd_buy_whl_tms",
                        "l12_mon_insu_buy_whl_tms","l12_mon_gld_buy_whl_tms","loan_act_ind","pl_crd_lmt_cd","ovd_30d_loan_tot_cnt",
                        "his_lng_ovd_day","hld_crd_card_grd_cd","crd_card_act_ind","l1y_crd_card_csm_amt_dlm_cd",
                        "atdd_type"]
    #lr_predict(data, category_feature, continuous_feature)
    # gbdt_predict(data, category_feature, continuous_feature)
    #gbdt_lr_predict(data, category_feature, continuous_feature)
    category_feature = gbdt_ffm_predict(data, category_feature, continuous_feature)

    #data = pd.read_csv('data/data.csv')
    #df_train = pd.read_csv('data/train.csv')
    #FFMFormat(data, 'flag', 'data/', df_train.shape[0], category_feature, continuous_feature)
