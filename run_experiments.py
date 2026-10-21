
"""
Machine Learning Assignment 2
Bagging, Boosting and Bias-Variance on Olivetti Faces
Student: Abhishek Kapar | Roll: M26ecm511
Seed: numeric digits of roll number = 26511

Run:
  python -m pip install -r requirements.txt
  python run_experiments.py

The script writes all measured results to outputs/ and plots to outputs/figures/.
It intentionally computes results rather than hard-coding or fabricating them.
"""
from pathlib import Path
import json, time, warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

SEED = 26511
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

def save_json(obj, path):
    def conv(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (np.bool_,)): return bool(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, indent=2, default=conv))

def load_data():
    data = fetch_olivetti_faces(shuffle=False)
    Ximg = data.images.astype(np.float32)
    y = data.target.astype(int)
    rng = np.random.default_rng(SEED)
    train, val, test = [], [], []
    # Split separately inside each identity, and save the resulting indices.
    for cls in range(40):
        ids = np.flatnonzero(y == cls)
        ids = rng.permutation(ids)
        train.extend(ids[:6]); val.extend(ids[6:8]); test.extend(ids[8:10])
    split = {"train": sorted(train), "validation": sorted(val), "test": sorted(test),
             "seed": SEED, "split_rule": "per identity: 6 train, 2 validation, 2 test"}
    split_path = OUT / "split.json"
    if split_path.exists():
        # Reuse the exact saved split on every run.
        saved = json.loads(split_path.read_text())
        tr, va, te = saved["train"], saved["validation"], saved["test"]
        assert len(tr)==240 and len(va)==80 and len(te)==80
        split = saved
    else:
        save_json(split, split_path)
        tr, va, te = split["train"], split["validation"], split["test"]
    X = Ximg.reshape(len(Ximg), -1)
    return X, y, np.array(tr), np.array(va), np.array(te), split

def masked(X):
    Z = X.reshape(-1,64,64).copy()
    Z[:,24:40,24:40] = 0.0
    return Z.reshape(-1,4096)

def scores(model, X, y):
    pred = model.predict(X)
    return float(accuracy_score(y,pred)), pred

def tree_factory(depth, leaf=1):
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                  random_state=SEED)

class ScratchBagging:
    """Bootstrap aggregation of DecisionTreeClassifier models; majority vote."""
    def __init__(self, n_estimators=10, max_depth=None, min_samples_leaf=1, seed=SEED):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.min_samples_leaf=min_samples_leaf; self.seed=seed
        self.estimators_=[]; self.inbag_=[]; self.classes_=None
    def fit(self,X,y):
        rng=np.random.default_rng(self.seed)
        n=len(y); self.classes_=np.unique(y)
        self.estimators_=[]; self.inbag_=[]
        for b in range(self.n_estimators):
            ids=rng.integers(0,n,size=n)
            model=DecisionTreeClassifier(max_depth=self.max_depth,
                  min_samples_leaf=self.min_samples_leaf, random_state=self.seed+b)
            model.fit(X[ids],y[ids])
            self.estimators_.append(model)
            self.inbag_.append(ids)
        return self
    def predict_votes(self,X):
        allpred=np.stack([m.predict(X) for m in self.estimators_],axis=0)
        votes=np.stack([(allpred==c).sum(axis=0) for c in self.classes_],axis=1)
        return votes
    def predict(self,X):
        votes=self.predict_votes(X)
        return self.classes_[np.argmax(votes,axis=1)]
    def predict_proba(self,X):
        votes=self.predict_votes(X).astype(float)
        return votes / len(self.estimators_)
    def oob_accuracy(self,X,y):
        n=len(y); class_to_col={c:i for i,c in enumerate(self.classes_)}
        vote=np.zeros((n,len(self.classes_)),dtype=float); count=np.zeros(n,dtype=int)
        for model, ids in zip(self.estimators_,self.inbag_):
            inbag=np.zeros(n,dtype=bool); inbag[np.unique(ids)]=True
            oob=np.flatnonzero(~inbag)
            if len(oob):
                pred=model.predict(X[oob])
                for idx,p in zip(oob,pred): vote[idx,class_to_col[p]]+=1
                count[oob]+=1
        valid=count>0
        pred=self.classes_[np.argmax(vote[valid],axis=1)]
        return float(accuracy_score(y[valid],pred)), int(valid.sum()), count

def pairwise_disagreement(models, X):
    preds=np.stack([m.predict(X) for m in models])
    vals=[]
    for i in range(len(models)):
        for j in range(i+1,len(models)):
            vals.append(np.mean(preds[i]!=preds[j]))
    return float(np.mean(vals)) if vals else 0.0

def mean_vote_margin(model, X):
    votes=model.predict_proba(X)
    sortedv=np.sort(votes,axis=1)
    return float(np.mean(sortedv[:,-1]-sortedv[:,-2]))

class ScratchSAMME:
    """Multiclass SAMME AdaBoost using weighted DecisionTreeClassifier learners."""
    def __init__(self, n_estimators=50, max_depth=1, seed=SEED):
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.seed=seed
        self.estimators_=[]; self.estimator_weights_=[]; self.errors_=[]
        self.sample_weight_history_=[]; self.weighted_train_error_history_=[]
        self.max_sample_weight_history_=[]; self.top10_weight_mass_history_=[]
        self.classes_=None
    def fit(self,X,y):
        n=len(y); self.classes_=np.unique(y); K=len(self.classes_)
        w=np.ones(n)/n
        self.estimators_=[]; self.estimator_weights_=[]; self.errors_=[]
        self.sample_weight_history_=[]; self.weighted_train_error_history_=[]
        self.max_sample_weight_history_=[]; self.top10_weight_mass_history_=[]
        for t in range(self.n_estimators):
            learner=DecisionTreeClassifier(max_depth=self.max_depth,
                    random_state=self.seed+t)
            learner.fit(X,y,sample_weight=w)
            pred=learner.predict(X)
            err=float(np.sum(w*(pred!=y)))
            # Save round-level diagnostics using the weights actually used to fit this learner.
            self.sample_weight_history_.append(w.copy())
            self.weighted_train_error_history_.append(err)
            self.max_sample_weight_history_.append(float(np.max(w)))
            self.top10_weight_mass_history_.append(float(np.sort(w)[-min(10,n):].sum()))
            # A learner no better than random multiclass prediction is not accepted.
            if err >= 1.0 - 1.0/K - 1e-12:
                if not self.estimators_:
                    # Keep explicit record and stop: no valid weak learner exists.
                    self.errors_.append(err)
                break
            err=max(err,1e-12)
            alpha=float(np.log((1-err)/err)+np.log(K-1))
            self.estimators_.append(learner)
            self.estimator_weights_.append(alpha)
            self.errors_.append(err)
            w *= np.exp(alpha*(pred!=y))
            w /= w.sum()
            if err <= 1e-12: break
        return self
    def decision_function(self,X):
        scores=np.zeros((len(X),len(self.classes_)),float)
        cmap={c:i for i,c in enumerate(self.classes_)}
        for h,a in zip(self.estimators_,self.estimator_weights_):
            p=h.predict(X)
            for i,c in enumerate(p): scores[i,cmap[c]] += a
        return scores
    def predict(self,X):
        if not self.estimators_:
            return np.repeat(self.classes_[0],len(X))
        return self.classes_[np.argmax(self.decision_function(X),axis=1)]
    def predict_proba(self,X):
        s=self.decision_function(X)
        # Assignment requests normalized weighted vote q_k = S_k / sum_t alpha_t.
        den=sum(self.estimator_weights_)
        if den <= 0: return np.ones_like(s)/s.shape[1]
        return s/den

def experiment_q1(X,y,tr,va,te):
    Xtr,ytr=X[tr],y[tr]; Xva,yva=X[va],y[va]; Xte,yte=X[te],y[te]
    Xmask=masked(Xte)
    depths=[2,4,8,16,None]; leaves=[1,2,4]
    rows=[]; models={}
    for d in depths:
        for leaf in leaves:
            m=tree_factory(d,leaf).fit(Xtr,ytr)
            a,b,c=[scores(m,A,B)[0] for A,B in [(Xtr,ytr),(Xva,yva),(Xte,yte)]]
            rows.append({"max_depth":"None" if d is None else d,"min_samples_leaf":leaf,
                         "train_accuracy":a,"validation_accuracy":b,"clean_test_accuracy":c})
            models[(d,leaf)]=m
    # Validation accuracy only; deterministic tie-break: shallower depth, then larger leaf.
    best=sorted(rows,key=lambda r:(-r["validation_accuracy"],
              (10**6 if r["max_depth"]=="None" else r["max_depth"]),-r["min_samples_leaf"]))[0]
    d=None if best["max_depth"]=="None" else best["max_depth"]; leaf=best["min_samples_leaf"]
    tree=models[(d,leaf)]
    save_json({"grid":rows,"selected":best},OUT/"q1a_tree_grid.json")
    bagrows=[]; bagmodels={}
    for B in [1,5,10,25,50,100,200]:
        tic=time.perf_counter()
        bag=ScratchBagging(B,d,leaf).fit(Xtr,ytr)
        elapsed=time.perf_counter()-tic
        tracc=scores(bag,Xtr,ytr)[0]; vaacc=scores(bag,Xva,yva)[0]; teacc=scores(bag,Xte,yte)[0]
        oob,nvalid,counts=bag.oob_accuracy(Xtr,ytr)
        unique_counts=[len(np.unique(ids)) for ids in bag.inbag_]
        n_train=len(Xtr)
        expected_unique=n_train*(1-(1-1/n_train)**n_train)
        bagrows.append({"B":B,"train_accuracy":tracc,"validation_accuracy":vaacc,
                        "clean_test_accuracy":teacc,"oob_accuracy":oob,
                        "oob_examples_with_votes":nvalid,"mean_oob_trees_per_training_example":float(np.mean(counts)),
                        "mean_distinct_examples_per_bootstrap_sample":float(np.mean(unique_counts)),
                        "theoretical_expected_distinct_examples":float(expected_unique),
                        "expected_oob_trees":B*(1-1/n_train)**n_train,"training_seconds":elapsed})
        bagmodels[B]=bag
    save_json({"results":bagrows},OUT/"q1b_bagging.json")
    # Tree disagreement and margin measured for final selected B by validation accuracy.
    bestB=sorted(bagrows,key=lambda r:(-r["validation_accuracy"],r["B"]))[0]["B"]
    bag=bagmodels[bestB]
    # Each fitted tree is a bagging learner; disagreement on clean test.
    dis=pairwise_disagreement(bag.estimators_,Xte)
    margin=mean_vote_margin(bag,Xte)
    oobrows=[]
    for B in [1,5,10,25,50,100,200]:
        bm=bagmodels[B]
        oobacc,_,_=bm.oob_accuracy(Xtr,ytr)
        oobrows.append({"B":B,"oob_accuracy":oobacc,
                        "pairwise_disagreement":pairwise_disagreement(bm.estimators_,Xte),
                        "mean_vote_margin":mean_vote_margin(bm,Xte)})
    save_json({"selected_B":bestB,"selected_depth":d,"selected_leaf":leaf,
               "selected_test_pairwise_disagreement":dis,"selected_test_mean_vote_margin":margin,
               "curves":oobrows},OUT/"q1c_ensemble_diagnostics.json")
    # Single tree vs final bagged model on clean and masked test; confidence = top normalized vote.
    tree_clean,treepred=scores(tree,Xte,yte); tree_mask,treepredm=scores(tree,Xmask,yte)
    bag_clean,bagpred=scores(bag,Xte,yte); bag_mask,bagpredm=scores(bag,Xmask,yte)
    probs_tree=np.zeros((len(Xte),40)); probs_tree[np.arange(len(Xte)),tree.predict(Xte)]=1
    probs_bag=bag.predict_proba(Xte)
    probs_bagm=bag.predict_proba(Xmask)
    examples=[]
    for i in range(len(Xte)):
        if (treepred[i]!=yte[i] and bagpred[i]==yte[i]) or (treepred[i]==yte[i] and bagpred[i]!=yte[i]):
            examples.append({"test_index":int(te[i]),"true_label":int(yte[i]),
             "tree_prediction":int(treepred[i]),"bagging_prediction":int(bagpred[i]),
             "tree_vote_counts":{str(int(treepred[i])):1},
             "bag_vote_counts":{str(int(k)):int(v) for k,v in zip(bag.classes_,bag.predict_votes(Xte[i:i+1])[0])},
             "tree_correct":bool(treepred[i]==yte[i]),"bag_correct":bool(bagpred[i]==yte[i])})
        if len(examples)>=4: break
    save_json({"selected_tree":{"depth":d,"min_samples_leaf":leaf},
      "selected_bagging_B":bestB,
      "clean_test":{"tree_accuracy":tree_clean,"bagging_accuracy":bag_clean},
      "masked_test":{"tree_accuracy":tree_mask,"bagging_accuracy":bag_mask},
      "mean_max_confidence":{"tree_clean":float(np.mean(np.max(probs_tree,axis=1))),
       "bag_clean":float(np.mean(np.max(probs_bag,axis=1))),
       "bag_masked":float(np.mean(np.max(probs_bagm,axis=1)))},
      "disagreement_examples":examples},OUT/"q1d_clean_masked.json")
    return {"tree":tree,"bag":bag,"depth":d,"leaf":leaf,"B":bestB,"bag_rows":bagrows}

def experiment_q2(X,y,tr,va,te, q1):
    Xtr,ytr=X[tr],y[tr]; Xva,yva=X[va],y[va]; Xte,yte=X[te],y[te]
    # Small hand-auditable 3-class example (10 samples) and one complete SAMME update.
    toyX=np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]],float)
    toyy=np.array([0,0,0,0,1,1,1,2,2,2])
    w=np.ones(10)/10
    toy=DecisionTreeClassifier(max_depth=1,random_state=SEED).fit(toyX,toyy,sample_weight=w)
    p=toy.predict(toyX); err=float(np.sum(w*(p!=toyy))); K=3
    toy_alpha=float(np.log((1-err)/err)+np.log(K-1)) if err < 1-1/K else None
    wnew=w.copy()
    if toy_alpha is not None:
        wnew *= np.exp(toy_alpha*(p!=toyy)); wnew/=wnew.sum()
    save_json({"toy_X":toyX.ravel(),"toy_y":toyy,"initial_weights":w,
       "weak_predictions":p,"weighted_error":err,"K":K,"alpha":toy_alpha,
       "updated_normalized_weights":wnew,"updated_weight_sum":float(wnew.sum())},OUT/"q2a_samme_toy.json")
    rows=[]; models={}
    for depth in [1,2,3,4]:
        for rounds in [1,2,3,4,10,25,50,100]:
            tic=time.perf_counter()
            m=ScratchSAMME(rounds,depth).fit(Xtr,ytr)
            elapsed=time.perf_counter()-tic
            tracc=scores(m,Xtr,ytr)[0]; vaacc=scores(m,Xva,yva)[0]; teacc=scores(m,Xte,yte)[0]
            rows.append({"max_depth":depth,"requested_rounds":rounds,"actual_learners":len(m.estimators_),
                "train_accuracy":tracc,"validation_accuracy":vaacc,"clean_test_accuracy":teacc,
                "training_seconds":elapsed,"final_learner_weight":m.estimator_weights_[-1] if m.estimator_weights_ else None,
                "weighted_training_errors":m.weighted_train_error_history_,
                "learner_weights":m.estimator_weights_,
                "max_sample_weights":m.max_sample_weight_history_,
                "top10_sample_weight_mass":m.top10_weight_mass_history_})
            models[(depth,rounds)]=m
    bestrow=sorted(rows,key=lambda r:(-r["validation_accuracy"],r["max_depth"],r["requested_rounds"]))[0]
    best=models[(bestrow["max_depth"],bestrow["requested_rounds"])]
    save_json({"grid":rows,"selected":bestrow,
      "selected_learner_weights":best.estimator_weights_,"selected_weighted_errors":best.errors_,
      "selected_round_diagnostics":{"weighted_training_error":best.weighted_train_error_history_,
       "learner_weight":best.estimator_weights_,"max_sample_weight":best.max_sample_weight_history_,
       "top10_sample_weight_mass":best.top10_weight_mass_history_}},OUT/"q2b_boosting_grid.json")
    # Q2(c): compare 25, 50, 100 learners for the selected depth.
    compare=[]
    for rounds in [25,50,100]:
        m=ScratchSAMME(rounds,bestrow["max_depth"]).fit(Xtr,ytr)
        compare.append({"learners_requested":rounds,"learners_actual":len(m.estimators_),
          "train_accuracy":scores(m,Xtr,ytr)[0],"validation_accuracy":scores(m,Xva,yva)[0],
          "clean_test_accuracy":scores(m,Xte,yte)[0],
          "training_seconds":next(r["training_seconds"] for r in rows if r["max_depth"]==bestrow["max_depth"] and r["requested_rounds"]==rounds)})
    # Show a high-weight training sample and its model prediction.
    weights=np.ones(len(ytr))/len(ytr)
    learner=DecisionTreeClassifier(max_depth=bestrow["max_depth"],random_state=SEED)
    # Recover final stage sample weights by replaying the selected boosting process.
    replay=ScratchSAMME(bestrow["requested_rounds"],bestrow["max_depth"]).fit(Xtr,ytr)
    # Identify the sample carrying the largest final-stage weight, and report its true/predicted class.
    final_weights=(replay.sample_weight_history_[-1] if replay.sample_weight_history_ else np.ones(len(ytr))/len(ytr))
    highidx=int(np.argmax(final_weights))
    error_counts=np.zeros(len(ytr),int)
    for h in replay.estimators_: error_counts += (h.predict(Xtr)!=ytr)
    save_json({"comparison_25_50_100":compare,
      "high_weight_training_sample":{"training_position":highidx,"original_dataset_index":int(tr[highidx]),
       "true_label":int(ytr[highidx]),"prediction":int(replay.predict(Xtr[highidx:highidx+1])[0]),
       "final_stage_sample_weight":float(final_weights[highidx]),
       "number_of_learner_errors":int(error_counts[highidx])}},OUT/"q2c_comparison.json")
    # Compare final selected tree, bagging and boosting clean/masked.
    Xmask=masked(Xte)
    tree=q1["tree"]; bag=q1["bag"]
    models_eval={"single_tree":tree,"bagging":bag,"boosting":best}
    out={}
    for name,m in models_eval.items():
        cleanacc,cp=scores(m,Xte,yte); maskacc,mp=scores(m,Xmask,yte)
        pc=m.predict_proba(Xte); pm=m.predict_proba(Xmask)
        out[name]={"clean_accuracy":cleanacc,"masked_accuracy":maskacc,
           "mean_max_normalized_vote":float(np.mean(np.max(pc,axis=1))),
           "masked_mean_max_normalized_vote":float(np.mean(np.max(pm,axis=1))),
           "clean_predictions":cp,"masked_predictions":mp}
    save_json({"selected_boosting":{"depth":bestrow["max_depth"],"rounds":bestrow["requested_rounds"],
        "actual_learners":len(best.estimators_)},"evaluation":out},OUT/"q2d_clean_masked.json")
    return {"boost":best,"depth":bestrow["max_depth"],"rounds":bestrow["requested_rounds"],"rows":rows}

def prob_onehot(pred, classes):
    idx={c:i for i,c in enumerate(classes)}
    out=np.zeros((len(pred),len(classes)))
    for i,p in enumerate(pred): out[i,idx[p]]=1.0
    return out

def experiment_q3(X,y,tr,va,te,q1,q2):
    Xte,yte=X[te],y[te]; classes=np.unique(y); K=len(classes)
    # Keep clean test fixed. Each identity's six original training examples are sampled
    # with replacement to produce a new 240-example training set.
    Xoriginal=X[tr]; yoriginal=y[tr]
    identity_train={c:np.flatnonzero(yoriginal==c) for c in classes}
    model_specs=[("single_tree",q1["depth"],q1["leaf"]),
                 ("bagging",q1["depth"],q1["leaf"]),
                 ("boosting",q2["depth"],q2["rounds"])]
    predictions={name:[] for name,_,_ in model_specs}; accuracies={name:[] for name,_,_ in model_specs}
    for r in range(30):
        rng=np.random.default_rng(SEED+1000+r)
        sampled=[]
        for c in classes:
            ids=identity_train[c]
            sampled.extend(rng.choice(ids,size=6,replace=True))
        sampled=np.array(sampled)
        xr,yr=Xoriginal[sampled],yoriginal[sampled]
        for name,depth,param in model_specs:
            if name=="single_tree":
                m=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=param,random_state=SEED+r).fit(xr,yr)
            elif name=="bagging":
                m=ScratchBagging(100,depth,param,seed=SEED+r).fit(xr,yr)
            else:
                m=ScratchSAMME(param,depth,seed=SEED+r).fit(xr,yr)
            # Probability-like class vector: tree is one-hot; ensembles use normalized votes.
            p=m.predict_proba(Xte) if hasattr(m,"predict_proba") else prob_onehot(m.predict(Xte),classes)
            # Ensure column order matches classes.
            if hasattr(m,"classes_") and not np.array_equal(m.classes_,classes):
                fixed=np.zeros((len(Xte),K))
                for j,c in enumerate(m.classes_): fixed[:,np.where(classes==c)[0][0]]=p[:,j]
                p=fixed
            predictions[name].append(p)
            accuracies[name].append(float(accuracy_score(yte,m.predict(Xte))))
    result={}
    for name,_,_ in model_specs:
        P=np.stack(predictions[name]) # R x N x K
        meanp=P.mean(axis=0)
        target=np.zeros_like(meanp)
        for i,c in enumerate(yte): target[i,np.where(classes==c)[0][0]]=1
        bias2=float(np.mean(np.sum((meanp-target)**2,axis=1)))
        variance=float(np.mean(np.sum((P-meanp[None,:,:])**2,axis=2)))
        result[name]={"bias_squared":bias2,"variance":variance,
          "test_accuracy_mean":float(np.mean(accuracies[name])),
          "test_accuracy_std":float(np.std(accuracies[name],ddof=1)),
          "test_accuracies":accuracies[name]}
    # Depth study: three single trees and 100-tree bagging at each depth.
    depthrows=[]
    for depth in [2,4,8,16]:
        accs=[]; treevars=[]; bagaccs=[]; bagvars=[]
        for r in range(30):
            rng=np.random.default_rng(SEED+2000+r)
            sampled=[]
            for c in classes:
                ids=identity_train[c]; sampled.extend(rng.choice(ids,size=6,replace=True))
            sampled=np.array(sampled); xr,yr=Xoriginal[sampled],yoriginal[sampled]
            t=DecisionTreeClassifier(max_depth=depth,random_state=SEED+r).fit(xr,yr)
            accs.append(accuracy_score(yte,t.predict(Xte)))
            b=ScratchBagging(100,depth,1,seed=SEED+r).fit(xr,yr)
            bagaccs.append(accuracy_score(yte,b.predict(Xte)))
            treevars.append(t.get_n_leaves())
            bagvars.append(pairwise_disagreement(b.estimators_,Xte))
        depthrows.append({"depth":depth,"single_tree_mean_test_accuracy":float(np.mean(accs)),
          "single_tree_std_test_accuracy":float(np.std(accs,ddof=1)),
          "mean_leaf_count":float(np.mean(treevars)),
          "bagging_100_mean_test_accuracy":float(np.mean(bagaccs)),
          "bagging_100_std_test_accuracy":float(np.std(bagaccs,ddof=1)),
          "bagging_pairwise_disagreement":float(np.mean(bagvars))})
    save_json({"bias_variance":result,"depth_study":depthrows},OUT/"q3_bias_variance.json")
    return result,depthrows

def make_plots():
    # Read measured results and generate requested figures.
    q1b=json.loads((OUT/"q1b_bagging.json").read_text())["results"]
    q1c=json.loads((OUT/"q1c_ensemble_diagnostics.json").read_text())["curves"]
    q2b=json.loads((OUT/"q2b_boosting_grid.json").read_text())
    q3=json.loads((OUT/"q3_bias_variance.json").read_text())
    def plot(xs,ys,xlab,ylab,title,name):
        plt.figure(figsize=(6,4)); plt.plot(xs,ys,marker="o")
        plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title); plt.grid(True,alpha=.3)
        plt.tight_layout(); plt.savefig(FIG/name,dpi=200); plt.close()
    plot([r["B"] for r in q1b],[r["oob_accuracy"] for r in q1b],"Number of trees B","OOB accuracy","Bagging: OOB accuracy vs B","q1_oob_accuracy.png")
    plot([r["B"] for r in q1c],[r["pairwise_disagreement"] for r in q1c],"Number of trees B","Pairwise disagreement","Bagging: disagreement vs B","q1_disagreement.png")
    plot([r["B"] for r in q1c],[r["mean_vote_margin"] for r in q1c],"Number of trees B","Mean vote margin","Bagging: vote margin vs B","q1_vote_margin.png")
    # Boosting training weighted error by learner count for selected depth.
    sel=q2b["selected"]; depth=sel["max_depth"]
    sub=[r for r in q2b["grid"] if r["max_depth"]==depth]
    plot([r["requested_rounds"] for r in sub],[r["train_accuracy"] for r in sub],"Requested rounds","Training accuracy","Boosting: training accuracy vs rounds","q2_training_accuracy.png")
    # Plot all round-level diagnostics from the selected boosting configuration.
    diag=q2b["selected_round_diagnostics"]
    for key,ylab,title,filename in [
        ("weighted_training_error","Weighted training error","SAMME weighted training error","q2_weighted_training_error.png"),
        ("learner_weight","Learner weight alpha","SAMME learner weights","q2_learner_weights.png"),
        ("max_sample_weight","Maximum sample weight","SAMME maximum sample weight","q2_max_sample_weight.png"),
        ("top10_sample_weight_mass","Mass of 10 highest-weight samples","SAMME top-10 sample weight mass","q2_top10_weight_mass.png")]:
        vals=diag[key]
        if vals: plot(list(range(1,len(vals)+1)),vals,"Boosting round",ylab,title,filename)
    bvd=q3["bias_variance"]
    names=list(bvd); plot(range(len(names)),[bvd[n]["bias_squared"] for n in names],"Model index (tree, bagging, boosting)","Bias squared","Measured bias squared","q3_bias_squared.png")
    plot(range(len(names)),[bvd[n]["variance"] for n in names],"Model index (tree, bagging, boosting)","Variance","Measured variance","q3_variance.png")
    save_json({"figures":sorted(p.name for p in FIG.glob("*.png"))},OUT/"figures_manifest.json")

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    X,y,tr,va,te,split=load_data()
    print("Split:",len(tr),len(va),len(te),"seed",SEED)
    q1=experiment_q1(X,y,tr,va,te)
    q2=experiment_q2(X,y,tr,va,te,q1)
    q3,_=experiment_q3(X,y,tr,va,te,q1,q2)
    make_plots()
    print("Finished. Results:",OUT)
    print("Selected tree:",q1["depth"],q1["leaf"],"selected bagging B:",q1["B"])
    print("Selected boosting:",q2["depth"],q2["rounds"])
    print("Q3 bias/variance:",json.dumps(q3,indent=2))

if __name__=="__main__":
    main()
