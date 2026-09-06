"""LLM-to-constraint adapter. No generated SQL or generated executable code."""
import json
import os
import re
from operations_service.planning import constraints_checked


def table_id(value):
    value=value.lower()
    if re.fullmatch(r"b\d{1,2}",value): return f"bac_{int(value[1:]):02d}"
    return value


def parse_goal(text, previous, settings=None):
    if not isinstance(text,str) or not text.strip() or len(text)>4000: raise ValueError("Enter a goal of 1–4000 characters")
    settings=settings or {}
    current=constraints_checked(previous)
    patch={}
    mode="Template parser · no LLM configured"
    warning=None
    key=settings.get("api_key") or os.getenv("OPS_LLM_API_KEY","")
    if key:
        try:
            prompt=("Extract ONLY explicitly requested constraint changes from the user's text as a JSON object. "
                    "Never drop existing exclusions or relax previous constraints unless explicitly requested. "
                    "Allowed fields: pit (main/entry/vip/blackjack/slots), horizon (15/30), max_wait (number minutes), "
                    "extra_staff (0..2, means additional staffing beyond available relief), protect_vip (boolean), "
                    "excluded (full merged array of table IDs), priority (wait/cost/theo). B08 means bac_08. "
                    "For unsupported instructions return {\"unsupported\":\"short reason in English\"}. "
                    "Return only JSON, no SQL, commands or forecast numbers. Current constraints: "+json.dumps(current))
            provider=settings.get("provider") or os.getenv("OPS_LLM_PROVIDER","OpenAI")
            model=settings.get("model") or os.getenv("OPS_LLM_MODEL","gpt-4o-mini")
            base=settings.get("base_url") or os.getenv("OPS_LLM_BASE_URL")
            if provider=="Claude":
                from anthropic import Anthropic
                args=dict(api_key=key,timeout=20,max_retries=0)
                if base: args["base_url"]=base
                response=Anthropic(**args).messages.create(model=model,max_tokens=700,system=prompt,messages=[{"role":"user","content":text}])
                raw="".join(b.text for b in response.content if b.type=="text")
            else:
                from openai import OpenAI
                args=dict(api_key=key,timeout=20,max_retries=0)
                if base: args["base_url"]=base
                response=OpenAI(**args).chat.completions.create(model=model,temperature=0,messages=[{"role":"system","content":prompt},{"role":"user","content":text}],response_format={"type":"json_object"})
                raw=response.choices[0].message.content
            raw=re.sub(r"^```(?:json)?\s*|\s*```$","",raw.strip())
            patch=json.loads(raw)
            if not isinstance(patch,dict): raise ValueError("Model returned invalid constraints")
            if "unsupported" in patch: raise ValueError(str(patch["unsupported"]))
            # Runtime schema remains authoritative, independent of the model's output.
            constraints_checked({**current,**patch})
            # Removing exclusions is a separate manual edit, not an implicit model decision.
            if "excluded" in patch:
                patch["excluded"]=sorted(set(current["excluded"]+patch["excluded"]))
            if current["protect_vip"] and patch.get("protect_vip") is False:
                if not re.search(r"(?:允许|可以|allow|can).{0,20}vip|vip.{0,20}(?:允许|可以|may|can)",text.lower()):
                    raise ValueError("VIP protection cannot be relaxed implicitly")
            if patch.get("extra_staff",current["extra_staff"])>current["extra_staff"]:
                if not re.search(r"增加|额外|additional|extra",text.lower()):
                    raise ValueError("Additional staffing cannot be authorized implicitly")
            mode=f"LLM · {provider} / {model}"
        except Exception:
            patch={}
            mode="Template parser · LLM unavailable"
            warning="LLM request failed or returned unsupported constraints. Using supported templates; review the structured constraints before generating a plan."
    lower=text.lower()
    recognized=False
    for words,pit in [(r"主厅|主场|main", "main"),(r"入门|entry","entry"),(r"二十一点|blackjack","blackjack"),(r"老虎机|slots","slots")]:
        if re.search(words,lower): patch["pit"]=pit; recognized=True
    if re.search(r"vip\s*(?:区域|区|pit).*(?:优化|optimi)|(?:优化|optimi).*vip",lower): patch["pit"]="vip"; recognized=True
    if re.search(r"半小时|30\s*(?:分钟|min)",lower): patch["horizon"]=30; recognized=True
    elif re.search(r"15\s*(?:分钟|min)|一刻钟",lower): patch["horizon"]=15; recognized=True
    match=re.search(r"(?:等待|wait)[^。,.，]{0,30}?(\d+(?:\.\d+)?|五|三|十)\s*(?:分钟|min)",lower)
    if match:
        patch["max_wait"]={"五":5,"三":3,"十":10}.get(match[1],float(match[1]) if match[1][0].isdigit() else 5); recognized=True
    if re.search(r"不(?:能)?增加(?:人手|人员)|no (?:extra|additional) (?:staff|dealer)|without.*(?:staff|dealer)",lower): patch["extra_staff"]=0; recognized=True
    match=re.search(r"(?:允许|增加|allow|add)[^。,.，]{0,12}?(一|两|\d)\s*(?:名|个)?\s*(?:荷官|员工|人手|dealer|staff)",lower)
    if match: patch["extra_staff"]={"一":1,"两":2}.get(match[1],int(match[1]) if match[1].isdigit() else 1); recognized=True
    if "vip" in lower and re.search(r"不变|保护|不调整|unchanged|protect|keep",lower): patch["protect_vip"]=True; recognized=True
    if re.search(r"优先.*(?:成本|cost)|minimi[sz]e.*cost|cost first",lower): patch["priority"]="cost"; recognized=True
    if re.search(r"优先.*(?:等待|wait)|wait first",lower): patch["priority"]="wait"; recognized=True
    if re.search(r"优先.*(?:theo|收益)|maximi[sz]e.*theo",lower): patch["priority"]="theo"; recognized=True
    ids=re.findall(r"\b(?:bac_(?:left_|vip_)?\d{2}|bj_\d{2}|slots_\d{2}|b\d{1,2})\b",lower)
    if ids and re.search(r"不考虑|排除|exclude|avoid|不要",lower):
        patch["excluded"]=sorted(set(current["excluded"]+[table_id(v) for v in ids])); recognized=True
    if not key and not recognized or mode.startswith("Template") and not recognized:
        raise ValueError("Unsupported template. Use the manual constraints form or configure an LLM. Try: For the next 30 minutes, keep the main floor wait within 5 minutes, with no additional staff. Keep VIP minimums unchanged.")
    merged=constraints_checked({**current,**patch})
    if mode.startswith("Template"):
        warning=(warning+" " if warning else "")+"Template mode recognizes only the displayed fields; other wording is not interpreted. Confirm every constraint below."
    return dict(constraints=merged,mode=mode,warning=warning,changes={k:v for k,v in merged.items() if v!=current[k]})
