import time, itertools, math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.optimize import minimize

st.set_page_config(page_title="ReliefQ | Emergency Resource Optimizer", page_icon="🌐", layout="wide")

st.markdown("""
<style>
:root { --navy:#132b4f; --teal:#16a6a1; }
.block-container {padding-top:1.5rem; max-width:1400px;}
[data-testid="stSidebar"] {background:#132b4f;}
[data-testid="stSidebar"] * {color:white !important;}
h1,h2,h3 {color:#132b4f;}
.metric-card {background:#f5f8fc;border:1px solid #e3eaf3;border-radius:14px;padding:18px 20px;}
.small-muted {color:#68778c;font-size:0.9rem;}
.stButton>button {background:#137f83;color:white;border-radius:9px;border:0;font-weight:600;}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state: st.session_state.page = "Operational Overview"
with st.sidebar:
    st.markdown("## ◈ ReliefQ")
    st.caption("EMERGENCY RESOURCE OPTIMIZATION")
    st.divider()
    page = st.radio("NAVIGATION", ["Operational Overview","Resource Allocation","Quantum vs Classical","About ReliefQ"], index=["Operational Overview","Resource Allocation","Quantum vs Classical","About ReliefQ"].index(st.session_state.page))
    st.session_state.page = page
    st.divider()
    st.caption("PROTOTYPE STATUS")
    st.markdown("🟢 Scenario simulator")
    st.markdown("🟡 QAOA runs locally when Qiskit is installed")
    st.caption("Demo data only · Not for live emergency decisions")

st.title("ReliefQ")
st.markdown("### Quantum-Assisted Emergency Relief Resource Optimization")
st.markdown("<div class='small-muted'>Plan limited relief supplies with urgency-aware optimization, then compare a QAOA candidate against an exact classical baseline.</div>", unsafe_allow_html=True)
st.write("")

with st.expander("Configure emergency scenario", expanded=True):
    c1,c2,c3 = st.columns(3)
    with c1:
        supply = st.number_input("Available food kits", min_value=1, max_value=10, value=4, step=1)
        name_a = st.text_input("Location A", "Village A")
        need_a = st.number_input("Location A need", min_value=0, max_value=5, value=3, step=1)
        urg_a = st.slider("Location A urgency",1,10,9)
    with c2:
        water = st.number_input("Available water packs (context only)", min_value=0, max_value=1000, value=0, step=10)
        name_b = st.text_input("Location B", "Village B")
        need_b = st.number_input("Location B need", min_value=0, max_value=5, value=3, step=1)
        urg_b = st.slider("Location B urgency",1,10,5)
    with c3:
        name_c = st.text_input("Location C (optional)", "Village C")
        need_c = st.number_input("Location C need", min_value=0, max_value=2, value=0, step=1)
        urg_c = st.slider("Location C urgency",1,10,7)
        st.caption("For this QAOA demo, total food-kit demand must be 10 or fewer.")

needs=[int(need_a),int(need_b),int(need_c)]
names=[name_a or "Location A",name_b or "Location B",name_c or "Location C"]
urg=[int(urg_a),int(urg_b),int(urg_c)]
demand=sum(needs)
target=min(int(supply),demand)
if demand>10:
    st.error("Reduce total demand to 10 or fewer kits for this statevector demo.")
    st.stop()
if demand==0:
    st.warning("Enter at least one requested kit.")
    st.stop()

units=[i for i,n in enumerate(needs) for _ in range(n)]
n=len(units)
benefits=np.array([urg[i] for i in units],dtype=float)

def bits_idx(idx,n):
    return np.array([(idx>>i)&1 for i in range(n)],dtype=int)
def utility(bits): return float(np.dot(benefits,np.asarray(bits,dtype=float)))
def cost(bits): return -utility(bits)+30.0*(int(sum(bits))-target)**2

@st.cache_data(show_spinner=False)
def classical_solve(needs_tuple, urg_tuple, supply_value):
    units0=[i for i,nn in enumerate(needs_tuple) for _ in range(nn)]
    target0=min(supply_value,sum(needs_tuple))
    b=np.array([urg_tuple[i] for i in units0],dtype=float)
    best_score=-1; best_bits=None
    start=time.perf_counter()
    for bits in itertools.product([0,1],repeat=len(units0)):
        if sum(bits)!=target0: continue
        score=float(np.dot(b,np.asarray(bits,dtype=float)))
        if score>best_score: best_score=score; best_bits=np.array(bits,dtype=int)
    return best_bits,best_score,time.perf_counter()-start

def qaoa_solve(n,target,depth=2):
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        from qiskit.circuit.library import DiagonalGate
    except Exception as e:
        return None, {"error":str(e)}
    dim=2**n
    costs=np.array([cost(bits_idx(i,n)) for i in range(dim)])
    utils=np.array([utility(bits_idx(i,n)) for i in range(dim)])
    def circuit(params):
        gammas=params[:depth]; betas=params[depth:]
        qc=QuantumCircuit(n); qc.h(range(n))
        for layer in range(depth):
            qc.append(DiagonalGate(np.exp(-1j*gammas[layer]*costs)),range(n))
            for q in range(n): qc.rx(2*betas[layer],q)
        return qc
    def expected(params):
        sv=Statevector.from_instruction(circuit(params))
        return float(np.dot(np.asarray(sv.probabilities()),costs))
    rng=np.random.default_rng(12)
    x0=np.r_[rng.uniform(0,np.pi,depth),rng.uniform(0,np.pi/2,depth)]
    start=time.perf_counter()
    result=minimize(expected,x0,method="COBYLA",options={"maxiter":55,"rhobeg":0.7,"tol":1e-3})
    sv=Statevector.from_instruction(circuit(result.x))
    probs=np.asarray(sv.probabilities())
    feasible=[]
    for i,p in enumerate(probs):
        bits=bits_idx(i,n)
        if int(sum(bits))==target: feasible.append((float(p),utility(bits),bits))
    if not feasible: return None, {"error":"No feasible state found in this run."}
    feasible.sort(key=lambda x:(x[0],x[1]),reverse=True)
    p,score,bits=feasible[0]
    return bits,{"score":score,"probability":p,"feasible_probability":sum(x[0] for x in feasible),
                 "expected_utility":float(np.dot(probs,utils)),"runtime":time.perf_counter()-start}

classic_bits,classic_score,classic_time=classical_solve(tuple(needs),tuple(urg),int(supply))
classic_alloc=[0]*len(needs)
for j,bit in enumerate(classic_bits):
    if bit: classic_alloc[units[j]]+=1

if page=="Operational Overview":
    st.subheader("Operational Overview")
    a,b,c,d=st.columns(4)
    a.metric("Available food kits",int(supply))
    b.metric("Total requested",demand)
    c.metric("Kits allocatable",target)
    d.metric("Locations in scenario",sum(1 for x in needs if x>0))
    st.write("")
    left,right=st.columns([1.5,1])
    with left:
        st.markdown("#### Recommended allocation · exact baseline")
        df=pd.DataFrame({"Location":names,"Need":needs,"Urgency":urg,"Recommended kits":classic_alloc})
        df=df[df["Need"]>0]
        st.dataframe(df,use_container_width=True,hide_index=True)
        fig=px.bar(df,x="Location",y=["Need","Recommended kits"],barmode="group",title="Demand vs recommended allocation")
        fig.update_layout(legend_title_text="",plot_bgcolor="white",paper_bgcolor="white")
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown("#### Urgency profile")
        udf=pd.DataFrame({"Location":[names[i] for i in range(3) if needs[i]>0],"Urgency":[urg[i] for i in range(3) if needs[i]>0]})
        fig2=px.bar(udf,x="Location",y="Urgency",range_y=[0,10],text="Urgency")
        fig2.update_layout(plot_bgcolor="white",paper_bgcolor="white",yaxis_title="Urgency (1–10)")
        st.plotly_chart(fig2,use_container_width=True)
        st.info("The exact baseline maximizes urgency-weighted kits delivered under the current supply constraint. This simple objective does not yet enforce fairness floors.")

elif page=="Resource Allocation":
    st.subheader("Resource Allocation")
    st.markdown("Change the scenario in **Configure emergency scenario** above, then review the exact allocation below.")
    df=pd.DataFrame({"Location":names,"Requested kits":needs,"Urgency":urg,"Allocated kits":classic_alloc})
    df["Unmet need"]=df["Requested kits"]-df["Allocated kits"]
    st.dataframe(df[df["Requested kits"]>0],use_container_width=True,hide_index=True)
    fig=px.bar(df[df["Requested kits"]>0],x="Location",y=["Allocated kits","Unmet need"],barmode="stack",title="Allocated vs unmet kits")
    fig.update_layout(plot_bgcolor="white",paper_bgcolor="white")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Current model: one resource type (food kits), urgency-weighted objective, and a fixed total allocation. Water packs are a future extension, not part of the optimization yet.")

elif page=="Quantum vs Classical":
    st.subheader("Quantum vs Classical")
    st.markdown("Run a small Qiskit QAOA statevector simulation and compare its selected feasible candidate with the exact classical optimum.")
    st.warning("Statevector QAOA runs on your computer's classical CPU; this is not quantum-hardware execution and does not demonstrate quantum advantage.")
    depth=st.select_slider("QAOA depth (p)",options=[1,2,3],value=2)
    if st.button("Run QAOA comparison",type="primary"):
        with st.spinner("Optimizing QAOA parameters; small instances may take a little time..."):
            qbits,qinfo=qaoa_solve(n,target,depth)
        if qbits is None:
            st.error("QAOA could not run. Install the packages from requirements.txt. Details: "+str(qinfo.get("error","unknown error")))
        else:
            qalloc=[0]*len(needs)
            for j,bit in enumerate(qbits):
                if bit: qalloc[units[j]]+=1
            qscore=qinfo["score"]
            gap=0 if classic_score==0 else 100*(classic_score-qscore)/classic_score
            x1,x2,x3,x4=st.columns(4)
            x1.metric("Exact classical score",f"{classic_score:.1f}")
            x2.metric("QAOA candidate score",f"{qscore:.1f}")
            x3.metric("Objective gap",f"{gap:.1f}%")
            x4.metric("QAOA feasible probability",f"{qinfo['feasible_probability']:.1%}")
            df=pd.DataFrame({"Location":names,"Need":needs,"Urgency":urg,"Classical":classic_alloc,"QAOA candidate":qalloc})
            df=df[df["Need"]>0]
            st.dataframe(df,use_container_width=True,hide_index=True)
            fig=px.bar(df,x="Location",y=["Classical","QAOA candidate"],barmode="group",title="Allocation comparison")
            fig.update_layout(plot_bgcolor="white",paper_bgcolor="white",legend_title_text="")
            st.plotly_chart(fig,use_container_width=True)
            st.write(f"Classical exact search runtime: `{classic_time:.4f} s`")
            st.write(f"QAOA simulation + parameter optimization runtime: `{qinfo['runtime']:.4f} s`")
            st.caption(f"Probability of the selected feasible bitstring: {qinfo['probability']:.3f}. Runtime is machine- and load-dependent; it is not a fair hardware speed comparison.")
            if gap < 1e-6: st.success("This QAOA candidate matches the exact baseline objective for this run.")
            else: st.info("The QAOA candidate is approximate and did not match the exact optimum in this run. That is a valid comparison result.")

else:
    st.subheader("About ReliefQ")
    st.markdown("""
**Problem:** Emergency response teams have limited supplies and multiple affected locations with different needs and urgency levels.

**Approach:** Formulate per-kit decisions as binary variables. Maximize urgency-weighted delivery value while allocating exactly the lesser of total supply and total demand.

**Classical baseline:** Exhaustive enumeration finds the exact optimum for this small instance.

**Quantum component:** Qiskit-based QAOA uses a cost-phase operator and mixer circuit; the resulting candidate is compared against the exact baseline.

**Limitations:** Demo data only; two or three locations, one optimized resource type, up to 10 requested kits. Fairness constraints, transport costs, multiple resource types, real disaster data, and production security are not included yet.
""")
st.divider()
st.caption("ReliefQ prototype · Educational decision-support demo · Validate all real-world allocations with relief professionals.")
