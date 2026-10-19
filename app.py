import io, math, re
from datetime import date
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="🌎", layout="wide")
st.title("🌎 Hispanic Heritage Historical Data Explorer")
st.caption("Explore World Bank historical indicators, fit polynomial models, and investigate functions at Algebra 2, Precalculus, or Calculus level.")
st.info("Starter research question: How have selected social and economic indicators changed over time across Latin American countries? Brazil is included as a Latin American comparison; Latin America contains countries with varied histories, languages, and identities.")

COUNTRIES = {"Mexico (MX)":"MEX","Brazil (BR)":"BRA","Argentina (AR)":"ARG","Colombia (CO)":"COL","Chile (CL)":"CHL","Peru (PE)":"PER","Venezuela (VE)":"VEN","Ecuador (EC)":"ECU","Uruguay (UY)":"URY"}
INDICATORS = {
"Life expectancy at birth":("SP.DYN.LE00.IN","years","Average number of years a newborn is expected to live under current mortality patterns.","Not a measure of an individual's lifespan."),
"Unemployment, national estimate":("SL.UEM.TOTL.NE.ZS","% of labor force","Unemployment as a percentage of the total labor force, national estimate.","A rate, not a count of unemployed people."),
"GDP per capita":("NY.GDP.PCAP.CD","current US$ per person","GDP divided by midyear population, in current US dollars.","Not average wealth or household income; affected by prices and exchange rates."),
"Crude birth rate":("SP.DYN.CBRT.IN","births per 1,000 people","Live births in a year per 1,000 people.","A rate, not a total birth count."),
"Net migration":("SM.POP.NETM","people","Net number of migrants during the period.","Not an emigration count; positive and negative values are possible."),
"Intentional homicide rate":("VC.IHR.PSRC.P5","per 100,000 people","Intentional homicides per 100,000 people.","Reporting and coverage may differ."),
"Consumer price inflation":("FP.CPI.TOTL.ZG","% annual change","Annual percentage change in the consumer price index.","A rate of price change, not the price level."),
"Central government debt":("GC.DOD.TOTL.GD.ZS","% of GDP","Central government debt as a percentage of GDP.","Not necessarily total public-sector debt."),
"Total reserves including gold":("FI.RES.TOTL.CD","current US$","Total reserves including gold, in current US dollars.","Amounts can change with valuation and exchange rates."),
"External balance on goods and services":("NE.RSB.GNFS.ZS","% of GDP","External balance on goods and services as a percentage of GDP.","Positive and negative values represent different balance positions.")
}
CONTEXTS = {
"Business":("revenue","time","US dollars","years","a business revenue model","a*x**2 + b*x + c"),
"Engineering":("displacement","time","meters","seconds","an engineering motion model","a*sin(b*x) + c"),
"Medical":("concentration","time","mg/L","hours","a medical concentration model","a*exp(-b*x) + c"),
"Sports science":("position","time","meters","seconds","a sports motion model","a*x**2 + b*x + c"),
"Biology":("population","time","organisms","days","a population model","a*exp(b*x)"),
"Bioengineering":("response","time","response units","hours","a bioengineering response model","a*sin(b*x) + c"),
"Economic forecasting":("index","time","index points","years","an economic index model","a*x**2 + b*x + c"),
"Weather":("temperature","time","°C","hours","a weather temperature model","a*sin(b*x) + c"),
"Custom":("y","x","y-units","x-units","a custom model","a*x**2 + b*x + c")
}
API = "https://api.worldbank.org/v2/country"

with st.sidebar:
    st.header("Data controls")
    country_label = st.selectbox("Featured country", list(COUNTRIES))
    country_code = COUNTRIES[country_label]
    indicator_label = st.selectbox("Indicator", list(INDICATORS))
    code, units, definition, caveat = INDICATORS[indicator_label]
    pathway = st.selectbox("Mathematics pathway", ["Honors Algebra 2","Precalculus","Calculus"])
    degree = st.slider("Polynomial degree",3,8,3)
    sample_step = st.slider("Graph sample increment (years)",1,10,3)
    horizon = st.slider("Forward display horizon (years)",0,50,10)
    compare = st.multiselect("Comparison countries",[c for c in COUNTRIES if c != country_label], default=["Brazil (BR)"] if country_label != "Brazil (BR)" else ["Mexico (MX)"])
    printer = st.checkbox("Printer-friendly report view")

yr_now = date.today().year
first, last = yr_now-70, yr_now-1
api_url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{code}?format=json"
st.subheader("Indicator and source")
st.write(f"**Definition:** {definition}")
st.write(f"**Units:** {units}")
st.write(f"**Interpretation note:** {caveat}")
st.markdown(f"**Official source:** [World Bank API]({api_url})")
st.caption(f"Requested years: {first}–{last}, inclusive. Retrieval date: {date.today().isoformat()}.")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch(iso3, indicator, start, end):
    url = f"{API}/{iso3}/indicator/{indicator}"
    try:
        r = requests.get(url, params={"date":f"{start}:{end}","format":"json","per_page":2000}, timeout=20)
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload,list) or len(payload)<2 or payload[1] is None:
            return pd.DataFrame(columns=["year","value"]), "The API returned no observations."
        rows = [{"year":int(item["date"]),"value":item["value"]} for item in payload[1] if item.get("date")]
        df = pd.DataFrame(rows)
        if df.empty: return pd.DataFrame(columns=["year","value"]), "The API response contained no year records."
        df["value"] = pd.to_numeric(df["value"],errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("year").drop_duplicates("year").reset_index(drop=True)
        return df, None
    except requests.Timeout: return pd.DataFrame(columns=["year","value"]), "Request timed out. Try again later."
    except (requests.RequestException, ValueError, TypeError, KeyError) as e: return pd.DataFrame(columns=["year","value"]), f"Could not retrieve World Bank data: {e}"

source, err = fetch(country_code,code,first,last)
if err: st.error(err)
if source.empty: st.warning("No usable observations were returned. No replacement or random values will be fabricated.")
else: st.success(f"{len(source)} non-missing observations; actual coverage {int(source.year.min())}–{int(source.year.max())}.")

def validate(df):
    issues=[]
    if df.empty: return ["The working table is empty."]
    if not {"year","value"}.issubset(df.columns): return ["Required columns are year and value."]
    years=pd.to_numeric(df.year,errors="coerce"); vals=pd.to_numeric(df.value,errors="coerce")
    if years.isna().any() or (years.dropna()%1!=0).any(): issues.append("Years must be integers.")
    if years.duplicated().any(): issues.append("Years must be unique.")
    if vals.isna().any() or not np.isfinite(vals.dropna()).all(): issues.append("Values must be finite numbers.")
    if ((years<1800)|(years>2200)).any(): issues.append("Years must be between 1800 and 2200.")
    if indicator_label=="Life expectancy at birth" and ((vals<0)|(vals>125)).any(): issues.append("Life expectancy should be 0–125 years.")
    if indicator_label=="Crude birth rate" and (vals<0).any(): issues.append("Crude birth rate cannot be negative.")
    return issues

key=f"{country_code}|{code}|{first}|{last}"
if st.session_state.get("data_key") != key:
    st.session_state.data_key=key
    st.session_state.source=source.copy()
    st.session_state.working=source.copy()
    st.session_state.edited=False
    st.session_state.single=None
    st.session_state.batch=None

st.subheader("Working data")
st.write("Original source data is kept separately from the editable scenario table.")
c1,c2=st.columns(2)
with c1:
    st.markdown("**Original source data**")
    st.dataframe(st.session_state.source,use_container_width=True,height=250)
with c2:
    st.markdown("**Editable year/value table**")
    edited=st.data_editor(st.session_state.working,num_rows="dynamic",use_container_width=True,key=f"editor_{key}")
    if st.button("Apply edits"):
        issues=validate(edited)
        if issues:
            for issue in issues: st.error(issue)
        else:
            st.session_state.working=edited.astype({"year":int,"value":float}).sort_values("year").reset_index(drop=True)
            st.session_state.edited=not st.session_state.working.equals(st.session_state.source.reset_index(drop=True))
            st.session_state.single=None; st.session_state.batch=None
            st.success("Validated edits applied.")
    if st.button("Reset to source"):
        st.session_state.working=st.session_state.source.copy().reset_index(drop=True)
        st.session_state.edited=False; st.session_state.single=None; st.session_state.batch=None
        st.rerun()

data=st.session_state.working.copy()
issues=validate(data) if not data.empty else ["No source data available."]
for issue in issues: st.warning(issue)
if st.session_state.edited: st.warning("Status: user-modified scenario; exports will label the data accordingly.")
valid=not issues and data.year.nunique()>=degree+1
if not valid and not issues: st.warning(f"Degree {degree} requires at least {degree+1} distinct years.")

def expr_text(coefs, base):
    terms=[]
    deg=len(coefs)-1
    for i,c in enumerate(coefs):
        p=deg-i
        if abs(c)<1e-12: continue
        terms.append(f"{c:.8g}" if p==0 else (f"{c:.8g}(year − {base:.3f})" if p==1 else f"{c:.8g}(year − {base:.3f})^{p}"))
    return " + ".join(terms).replace("+ -","− ")

def plot_model(data, model, base, low, high, horizon, step, single=None, batch=None, comparisons=None):
    fig,ax=plt.subplots(figsize=(10,5))
    ax.scatter(data.year,data.value,label="Observed / working data",s=28,zorder=4)
    xs=np.linspace(low,high+horizon,max(300,(high+horizon-low+1)*3))
    ax.plot(xs,model(xs-base),label="Polynomial fit / extrapolation",linewidth=2)
    ax.scatter(np.arange(low,high+1,step),model(np.arange(low,high+1,step)-base),marker="x",label=f"Model markers every {step} year(s)")
    ax.axvline(high,linestyle="--",label="Historical boundary")
    if single: ax.scatter([single["year"]],[single["estimate"]],marker="*",s=140,label="Single-year estimate")
    if batch is not None and not batch.empty: ax.scatter(batch.year,batch.estimate,marker="D",s=35,label="Batch estimates")
    for label,df in (comparisons or {}).items():
        if not df.empty: ax.plot(df.year,df.value,marker=".",linewidth=1,label=f"{label} observed")
    ax.set_xlabel("Year"); ax.set_ylabel(f"{indicator_label} ({units})"); ax.grid(True,alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    return fig

model=None; meta={}
if valid:
    years=data.year.to_numpy(float); vals=data.value.to_numpy(float); base=float(years.mean())
    model=np.poly1d(np.polyfit(years-base,vals,degree)); fitted=model(years-base); residual=vals-fitted
    ssres=float(np.sum(residual**2)); sstot=float(np.sum((vals-vals.mean())**2))
    r2=1-ssres/sstot if sstot>0 else float("nan"); mae=float(np.mean(np.abs(residual)))
    low,high=int(years.min()),int(years.max())
    meta={"base":base,"low":low,"high":high,"years":years,"vals":vals,"fitted":fitted,"residual":residual,"r2":r2,"mae":mae,"mean":float(vals.mean())}
    st.subheader("Polynomial model")
    equation=expr_text(model.c,base)
    st.markdown(f"**Fitted equation:** `f(year) = {equation}`")
    st.caption(f"Centered time t = year − {base:.6f}. Coefficients use centered time.")
    m=st.columns(5)
    for col,label,value in zip(m,["Degree","R² (in-sample)","MAE (in-sample)","Observations","Mean value"],[str(degree),f"{r2:.4f}" if math.isfinite(r2) else "undefined",f"{mae:.4g} {units}",str(len(data)),f"{vals.mean():.4g} {units}"]): col.metric(label,value)
    residuals=pd.DataFrame({"year":years.astype(int),"observed":vals,"fitted":fitted,"residual":residual})
    st.write("Residual = observed minus fitted. Positive residuals are above the curve.")
    st.dataframe(residuals,use_container_width=True,height=180)
    holdout=max(1,round(len(data)*.2))
    if len(data)-holdout>=degree+1:
        ordered=data.sort_values("year"); train,test=ordered.iloc[:-holdout],ordered.iloc[-holdout:]
        hb=float(train.year.mean()); hm=np.poly1d(np.polyfit(train.year-hb,train.value,degree))
        st.write(f"**Chronological holdout MAE:** {np.mean(np.abs(test.value-hm(test.year-hb))):.4g} {units}, measured on the final {holdout} year(s).")
    else: st.caption("Chronological holdout score unavailable: not enough training observations for this degree.")
    st.subheader("Calculate a value")
    left,right=st.columns(2)
    with left:
        single_year=st.number_input("Single year (1950–2100)",1950,2100,max(1950,min(2100,high)),1)
        if st.button("Calculate Value"):
            est=float(model(single_year-base))
            st.session_state.single={"year":int(single_year),"estimate":est,"classification":"within fitted range" if low<=single_year<=high else "outside fitted range"}
            st.session_state.batch=None
    with right:
        mode=st.radio("Batch input",["Year range","Custom years"],horizontal=True)
        step=st.slider("Year range step",1,10,1)
        a,b=st.columns(2)
        with a: rs=st.number_input("Range start",1950,2100,2000,1)
        with b: re_=st.number_input("Range end",1950,2100,2020,1)
        custom=st.text_input("Comma-separated years","2000, 2010, 2020",disabled=mode!="Custom years")
        if st.button("Calculate batch"):
            requested=[]; errors=[]
            if mode=="Year range":
                if re_<rs: errors.append("Range end must be at least range start.")
                else: requested=list(range(int(rs),int(re_)+1,step))
            else:
                for p in custom.split(","):
                    p=p.strip()
                    if not re.fullmatch(r"\d{4}",p): errors.append(f"Invalid year: {p}")
                    elif not 1950<=int(p)<=2100: errors.append(f"Year outside 1950–2100: {p}")
                    else: requested.append(int(p))
            if errors:
                for e in errors: st.error(e)
            elif not requested: st.error("Enter at least one valid year.")
            else:
                batch=pd.DataFrame({"year":requested})
                batch["estimate"]=[float(model(v-base)) for v in requested]
                batch["classification"]=batch.year.apply(lambda v:"within fitted range" if low<=v<=high else "outside fitted range")
                st.session_state.batch=batch; st.session_state.single=None
    single=st.session_state.single; batch=st.session_state.batch
    if single:
        st.success(f"{single['year']}: {single['estimate']:.7g} {units} ({single['classification']}). This is a fitted estimate, not necessarily an observed value.")
        if single["classification"]=="outside fitted range": st.warning("Extrapolation can be implausible; estimates are not silently clipped.")
    if batch is not None:
        st.dataframe(batch,use_container_width=True)
        if (batch.classification=="outside fitted range").any(): st.warning("Some batch years are extrapolations.")
    comparisons={}
    for label in compare:
        df,e=fetch(COUNTRIES[label],code,first,last)
        if e: st.caption(f"{label}: {e}")
        else:
            comparisons[label]=df
            st.caption(f"{label}: {len(df)} observations; coverage {int(df.year.min()) if not df.empty else 'n/a'}–{int(df.year.max()) if not df.empty else 'n/a'}. Gaps are not imputed.")
    fig=plot_model(data,model,base,low,high,horizon,sample_step,single,batch,comparisons)
    st.pyplot(fig,use_container_width=True)
    img=io.BytesIO(); fig.savefig(img,format="png",dpi=140,bbox_inches="tight"); img.seek(0)

    st.subheader("Course-level interpretation")
    if pathway=="Honors Algebra 2":
        st.write("The polynomial is mathematically defined for all real years, but the data support is strongest inside the observed range. Compare endpoint values to calculate average rate of change in indicator units per year. The leading term controls far-end behavior.")
    elif pathway=="Precalculus":
        st.write("Degree controls the maximum possible number of turning points and broad end behavior. Coefficient size and sign shape the curve. Compare average rates across intervals to describe how the trend rate changes; curve shape alone does not prove a historical cause.")
    else:
        st.write("For centered time t = year − base year, the first derivative estimates change in indicator units per year and the second derivative estimates change in that rate per year². These are properties of the fitted model, not causal explanations.")
    t=sp.symbols("t",real=True)
    fexpr=sum(sp.Float(float(c),12)*t**(degree-i) for i,c in enumerate(model.c))
    d1=sp.diff(fexpr,t); d2=sp.diff(fexpr,t,2)
    st.markdown(f"**f(t):** `{sp.sstr(fexpr)}`")
    st.markdown(f"**f′(t):** `{sp.sstr(d1)}`")
    st.markdown(f"**f″(t):** `{sp.sstr(d2)}`")
    if pathway=="Calculus":
        try:
            roots=[float(sp.re(z)) for z in sp.nroots(d1) if abs(float(sp.im(z)))<1e-8 and low-base<=float(sp.re(z))<=high-base]
        except Exception: roots=[]
        st.write("**Critical points in observed interval:**")
        if not roots: st.write("No real derivative roots detected in the observed interval.")
        for rt in sorted(roots):
            yr=rt+base; l=float(model(rt-1e-3,1)); r=float(model(rt+1e-3,1))
            typ="local minimum" if l<0<r else "local maximum" if l>0>r else "stationary point not classified as an extremum"
            st.write(f"- Year ≈ {yr:.4f}; fitted value ≈ {float(model(rt)):.6g}: {typ}.")
        try:
            roots2=[float(sp.re(z)) for z in sp.nroots(d2) if abs(float(sp.im(z)))<1e-8 and low-base<=float(sp.re(z))<=high-base]
        except Exception: roots2=[]
        st.write("**Potential inflection points:**")
        for rt in sorted(roots2):
            change=float(model(rt-1e-3,2))*float(model(rt+1e-3,2))<0
            st.write(f"- Year ≈ {rt+base:.4f}: " + ("concavity changes nearby." if change else "f″ is zero, but concavity change was not confirmed."))
        if not roots2: st.write("No real second-derivative roots detected in the observed interval.")
    st.caption(f"MAE ≈ {mae:.4g} {units}; it is not a validated 95% prediction interval. Polynomial extrapolation can be unreliable.")

    export=data.copy(); export["fitted_value"]=fitted; export["residual"]=residual
    results=batch.copy() if batch is not None else pd.DataFrame([single]) if single else pd.DataFrame(columns=["year","estimate","classification"])
    st.download_button("Download CSV",export.to_csv(index=False).encode(), "historical_data_explorer.csv","text/csv")
    xlsx=io.BytesIO()
    with pd.ExcelWriter(xlsx,engine="xlsxwriter") as writer:
        pd.DataFrame([["Country",country_label],["Indicator",indicator_label],["Code",code],["Units",units],["Definition",definition],["Source URL",api_url],["Retrieval date",date.today().isoformat()],["Data status","User-modified scenario" if st.session_state.edited else "Source-derived"],["Degree",degree],["Centered base year",base],["R² in-sample",r2],["MAE in-sample",mae]],columns=["Field","Value"]).to_excel(writer,sheet_name="Summary",index=False)
        export.to_excel(writer,sheet_name="Working Data",index=False); results.to_excel(writer,sheet_name="Calculated Years",index=False)
    st.download_button("Download Excel workbook",xlsx.getvalue(),"historical_data_explorer.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    pdf=io.BytesIO(); styles=getSampleStyleSheet()
    story=[Paragraph("Hispanic Heritage Historical Data Explorer",styles["Title"]),Paragraph(f"{country_label} — {indicator_label}",styles["Heading2"]),Paragraph(f"Code: {code}; units: {units}",styles["BodyText"]),Paragraph(definition,styles["BodyText"]),Paragraph(f"Source: {api_url}",styles["BodyText"]),Paragraph(f"Retrieved: {date.today().isoformat()}; status: {'user-modified scenario' if st.session_state.edited else 'source-derived'}",styles["BodyText"]),Spacer(1,8),Paragraph(f"Model: f(year) = {equation}",styles["BodyText"]),Paragraph(f"Degree {degree}; base year {base:.4f}; R² in-sample {r2:.5f}; MAE {mae:.5g}; n={len(data)}. MAE is not a prediction interval; extrapolation is uncertain.",styles["BodyText"]),Image(img,width=650,height=310)]
    preview=export.head(10).round(5).values.tolist()
    story += [Spacer(1,8),Paragraph("Data preview",styles["Heading3"]),Table([list(export.columns)]+preview,repeatRows=1,style=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightblue),("GRID",(0,0),(-1,-1),.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))]
    if not results.empty: story += [Spacer(1,8),Paragraph("Calculated years",styles["Heading3"]),Table([list(results.columns)]+results.astype(str).values.tolist(),repeatRows=1,style=TableStyle([("GRID",(0,0),(-1,-1),.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),8)]))]
    SimpleDocTemplate(pdf,pagesize=landscape(letter)).build(story)
    st.download_button("Download PDF report",pdf.getvalue(),"historical_data_explorer.pdf","application/pdf")
    plt.close(fig)

st.divider()
st.subheader("Function playground and contextual simulation")
st.write("Use `x` for the independent variable. Supported notation includes polynomial operations and sin, cos, tan, exp, log, sqrt.")
context=st.selectbox("Context",list(CONTEXTS))
dep0,ind0,du0,iu0,desc0,suggested=CONTEXTS[context]
choice=st.radio("Function source",["Historical curve of best fit","Suggested function","Build with parameters","Enter my own function"],horizontal=True)
dep=st.text_input("Dependent variable",dep0); indep=st.text_input("Independent variable",ind0)
du=st.text_input("Dependent units",du0); iu=st.text_input("Independent units",iu0)
desc=st.text_input("Context description",desc0)
c=st.columns(2)
with c[0]: xlo=st.number_input("Analysis interval start",0.0,key="xlo")
with c[1]: xhi=st.number_input("Analysis interval end",10.0,key="xhi")
p=st.columns(3)
with p[0]: aa=st.number_input("Parameter a",1.0)
with p[1]: bb=st.number_input("Parameter b",0.2)
with p[2]: cc=st.number_input("Parameter c",0.0)
x=sp.symbols("x",real=True)
if choice=="Historical curve of best fit":
    fx=sum(sp.Float(float(co),12)*(x-base)**(degree-i) for i,co in enumerate(model.c)) if valid else None
elif choice in ("Suggested function","Build with parameters"):
    fx=sp.sympify(suggested,locals={"x":x,"a":sp.Float(aa),"b":sp.Float(bb),"c":sp.Float(cc),"sin":sp.sin,"cos":sp.cos,"tan":sp.tan,"exp":sp.exp,"log":sp.log,"sqrt":sp.sqrt})
else:
    own=st.text_input("Enter f(x)", "a*x**2 + b*x + c")
    try:
        if len(own)>180 or not re.fullmatch(r"[0-9a-zA-Z_+\-*/().,\s^]+",own): raise ValueError("Use numbers, x, a, b, c, arithmetic operators, and supported functions.")
        fx=sp.sympify(own.replace("^","**"),locals={"x":x,"a":sp.Float(aa),"b":sp.Float(bb),"c":sp.Float(cc),"sin":sp.sin,"cos":sp.cos,"tan":sp.tan,"exp":sp.exp,"log":sp.log,"sqrt":sp.sqrt})
    except Exception as e:
        fx=None; st.error(f"Function input error: {e}")
if fx is not None:
    dfx=sp.diff(fx,x); ddfx=sp.diff(fx,x,2)
    st.markdown(f"**f(x):** `{sp.sstr(fx)}`")
    st.markdown(f"**f′(x):** `{sp.sstr(dfx)}`")
    st.markdown(f"**f″(x):** `{sp.sstr(ddfx)}`")
    lowx=min(xlo,xhi); highx=max(xlo,xhi)
    if lowx==highx: highx=lowx+1
    point=st.slider("Current input x",float(lowx),float(highx),float((lowx+highx)/2),step=max((highx-lowx)/100,0.001))
    showf=st.checkbox("Show function",True); showd=st.checkbox("Show first derivative",False); showdd=st.checkbox("Show second derivative",False)
    autoplay=st.checkbox("Auto-play simulation",False); speed=st.slider("Playback speed",1,10,5,key="context_speed")
    try:
        fn=sp.lambdify(x,fx,modules=["numpy"]); dfn=sp.lambdify(x,dfx,modules=["numpy"]); ddfn=sp.lambdify(x,ddfx,modules=["numpy"])
        ypoint=float(np.asarray(fn(point))); slope=float(np.asarray(dfn(point))); curve=float(np.asarray(ddfn(point)))
        st.metric(f"{dep} at x={point:g}",f"{ypoint:.6g} {du}")
        st.write(f"**Instantaneous rate:** {slope:.6g} {du}/{iu}. **Second derivative:** {curve:.6g} {du}/{iu}².")
        if xhi!=xlo:
            ya,yb=float(np.asarray(fn(xlo))),float(np.asarray(fn(xhi))); avg=(yb-ya)/(xhi-xlo)
            st.write(f"**Average rate of change** on [{xlo:g}, {xhi:g}]: {avg:.6g} {du}/{iu}.")
        grid=np.linspace(lowx,highx,500)
        def ev(func,arr):
            with np.errstate(all="ignore"): out=np.asarray(func(arr),dtype=float)
            out[~np.isfinite(out)]=np.nan
            return out
        fig2,ax=plt.subplots(figsize=(9,4.5))
        if showf:
            ax.plot(grid,ev(fn,grid),label=f"{dep} f(x)")
            ax.scatter([point],[ypoint],label="Current point",zorder=5)
            ax.plot(grid,slope*(grid-point)+ypoint,"--",label="Tangent at current point")
        if showd: ax.plot(grid,ev(dfn,grid),label="First derivative")
        if showdd: ax.plot(grid,ev(ddfn,grid),label="Second derivative")
        ax.set_xlabel(f"{indep} ({iu})"); ax.set_ylabel("Value / rate (units vary)"); ax.grid(True,alpha=.25); ax.legend(); fig2.tight_layout()
        st.pyplot(fig2,use_container_width=True); plt.close(fig2)
        st.caption(f"{desc} Visual scale is illustrative, not a validated physical simulator.")
        if autoplay:
            import streamlit.components.v1 as components
            components.html(f"""<div style="border:1px solid #cbd5e1;border-radius:12px;padding:16px;font-family:Arial"><b>{context} simulation</b><div style="height:16px;background:#e2e8f0;border-radius:9px;overflow:hidden;margin-top:10px"><div id="bar" style="height:100%;width:10%;background:linear-gradient(90deg,#0ea5e9,#22c55e)"></div></div><div id="label" style="font-size:12px;margin-top:5px">Illustrative scale</div></div><script>let v=0,d=1;setInterval(()=>{{v+=d*{speed};if(v>=100||v<=0)d*=-1;document.getElementById('bar').style.width=v+'%';document.getElementById('label').textContent=v.toFixed(0)+'% of illustrative scale';}},120)</script>""",height=100)
        else: st.progress(0.5,text="Illustrative context scale")
    except Exception as e: st.error(f"Could not evaluate over this interval. Check domain restrictions such as log(x) requiring x > 0. Details: {e}")

st.divider()
st.subheader("Deployment and verification")
st.markdown("""
**Local:** `pip install -r requirements.txt` then `python -m streamlit run app.py`.

**Streamlit Community Cloud:** Upload `app.py` and `requirements.txt` to the root of a GitHub repository. Create an app in Streamlit Community Cloud, select the repository/branch, set main file path to `app.py`, and deploy.

**Render:** Build command `pip install -r requirements.txt`; start command `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`.

**Checks:** compare API data against source links; test edits and Reset to source; try duplicate/non-integer years and too few observations; compare single and batch results for the same year; test malformed custom functions and `log(x)` across zero; toggle graph curves and change context/units/interval; download and open all three export types; inspect deployment logs for missing dependencies or incorrect app path.

**Limitations:** API coverage differs by country/indicator and missing values are not invented. Polynomial fits describe data, not causes; extrapolations may be implausible. MAE is not a 95% prediction interval. Context functions are classroom examples, not validated medical, financial, engineering, or weather forecasts.
""")
