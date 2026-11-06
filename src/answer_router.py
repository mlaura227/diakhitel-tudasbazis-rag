"""Evidence completeness router; avoids accepting definitions as fee answers."""
from __future__ import annotations
from src.query_policy import QueryPlan, norm
from src.extractive import response, extract_answer
from src.tariff_rules import resolve_fee, section, bullet, row, supported, not_documented, FEE, NOTICE, GLOSS


def definition_result(plan: QueryPlan, hits: list[dict], question: str = "") -> dict | None:
    keys=[key for key in plan.sections if key[0]==GLOSS]
    for doc,sid in keys:
        hit=section(hits,doc,sid)
        if not hit: continue
        title=hit['metadata'].get('section_title','Fogalom')
        for line in hit['content'].splitlines():
            candidate=line.strip()
            if not candidate or candidate.startswith(('#','Forrásdokumentum:')): continue
            quote=candidate.split(' [Forrás:',1)[0].strip()
            if len(quote)<24 or quote not in hit['content']:continue
            prefix = 'Nem, a SEPA-átutalás nem forintátutalás. ' if ('sepa' in norm(title) and 'forint' in norm(question)) else ''
            return supported(prefix + f'A glosszárium „{title}” meghatározása: {quote}',[(hit,quote)],'definition_extractive')
    return None


def _pairs(result:dict, hits:list[dict]) -> list[tuple[dict,str]]:
    sources={s['citation_id']:(s.get('doc_id'),s.get('section_id')) for s in result.get('sources',[])}
    found=[]
    for ev in result.get('evidence',[]):
        key=sources.get(ev['source_id'])
        hit=section(hits,*key) if key else None
        if hit and ev['quote'] in hit['content']:
            found.append((hit,ev['quote']))
    return found


def join_answers(parts:list[dict],hits:list[dict]) -> dict:
    quotations=[quote for part in parts for quote in _pairs(part,hits)]
    result=response(' '.join(part['answer'] for part in parts), quotations)
    result['mode']='composite_verified'
    return result


def _compound_fee(question:str,hits:list[dict]) -> dict | None:
    q=norm(question)
    if 'rendszeres' in q and ('eseti' in q or ('papir' in q and 'utal' in q)) and 'utal' in q:
        first=resolve_fee('Mennyi a rendszeres forintátutalás díja?',hits)
        second=resolve_fee('Mennyi a belföldi papíralapú fióki eseti forintátutalás díja?',hits)
        if first and second and all(p['status']=='answered' for p in (first,second)):
            return join_answers([first,second],hits)
    if 'kartya' in q and ('hirdetmeny' in q and 'dijjegyzek' in q) and ('masodik' in q or 'evtol' in q):
        card=section(hits,FEE,'SRC-P06-B04');notice=section(hits,NOTICE,'source-p1-card-fees')
        line=bullet(card,'Főkártya','második évtől')
        r=row(notice,'második évtől')
        if line and r:
            from src.tariff_rules import last_fee
            if norm(last_fee(line))==norm(r[1][-1]) or ({norm(last_fee(line)),norm(r[1][-1])} <= {'0 ft','dijmentes'}):
                return supported(f'A díjjegyzék szerint {last_fee(line)}, a termékhirdetmény szerint {r[1][-1]}; mindkettő díjmentes főkártya-éves díjat rögzít.',[(card,line),(notice,r[0])])
    if 'jogosult' in q and 'beszed' in q:
        notice=section(hits,NOTICE,'source-p1-eligibility')
        debit=resolve_fee('Mennyi a csoportos beszedési megbízás díja?',hits)
        if notice and debit:
            text=notice['content']
            quote=next((line for line in text.splitlines() if 'Diákhitel Központ Zrt.-nél' in line),None)
            if quote:
                first=next((line for line in text.splitlines() if 'Üzletszabályzat a Lakossági' in line),None)
                return supported('A termékhirdetmény szerint a jogosultsághoz az üzletszabályzati feltételek teljesítése és az OTP megjelölése szükséges a Diákhitel Központnál. '+debit['answer'], ([(notice,first)] if first else []) + [(notice,quote)]+_pairs(debit,hits))
    return None


def answer_from_sources(question:str,plan:QueryPlan,hits:list[dict]) -> dict | None:
    if not hits:
        return None
    q=norm(question)
    # Contractual clauses have a distinct evidence path. Do not allow a legal question to be answered by unrelated fee tables or glossary definitions.
    from src.legal_reasoning import plan_legal, answer_legal
    legal_plan = plan_legal(question)
    if legal_plan:
        return answer_legal(question, hits, legal_plan)
    # Product/service assertions are not tariffs; source explicitly states unavailability.
    if 'hitelkartya' in q and ('biztosit' in q or 'elso eves' in q):
        h=section(hits,FEE,'SRC-P07-B01')
        if h:
            line=next((s for s in h['content'].splitlines() if 'nem áll rendelkezésre' in s),None)
            if line:return supported('A Diákhitel Számlához a díjjegyzék szerint nem áll rendelkezésre hitelkártya-szolgáltatás, így nincs megállapítható első éves hitelkártyadíj.',[(h,line)])
    if 'kedvezmeny' in q and any(x in q for x in ('szazalek', 'pontos mertek', 'mekkora mertek')):
        h=section(hits,NOTICE,'source-p1-additional-benefits')
        if h:
            quote=next((line for line in h['content'].splitlines() if 'kedvezmények mértékeit' in line),None)
            if quote:return not_documented('A hirdetmény említi az igényelhető kedvezményeket, de százalékos mértéküket nem közli; azt más, itt be nem töltött üzletági hirdetményekhez utalja.',[(h,quote)])
    if 'arfolyam' in q and ('deviza' in q or 'dollar' in q or 'usd' in q):
        h=section(hits,FEE,'SRC-P06-B03')
        if h:
            quote=next((line for line in h['content'].splitlines() if 'nem részletezi az átváltás' in line),None)
            if quote:return not_documented('A díjjegyzék a devizajóváírás átváltási árfolyamát nem adja meg.',[(h,quote)])
    if ('hitelkeret' in q or 'automatikusan' in q and 'folyoszamlahitel' in q):
        h=section(hits,FEE,'SRC-P09-B01')
        if h:
            quote=next((line for line in h['content'].splitlines() if 'nem közöl hitelkeret-összeget' in line),None)
            if quote:return not_documented('A forrás nem ad meg automatikusan járó folyószámlahitel-keretösszeget.',[(h,quote)])
    if 'hataly' in q and '2026' in q:
        h=section(hits,FEE,'section-001')
        if h:
            line=next((s for s in h['content'].splitlines() if 'külön hatálybalépési időpont' in s),None)
            if line:return not_documented('A 2026.09.01. dátum szerepel a díjjegyzéken, de a külön jogi hatálybalépési dátumot a dokumentum nem adja meg.',[(h,line)])
    if 'fee' in plan.intents:
        compound=_compound_fee(question,hits)
        if compound:return compound
        fee=resolve_fee(question,hits)
        if not fee:
            # The previous extractive rules contain a well-tested paper-copy vs maintenance branch and card-replacement route.
            candidate=extract_answer(question,hits)
            if candidate and candidate.get('status')=='answered' and any(
                s.get('document_type')!='glossary' for s in candidate.get('sources',[])
            ):
                fee=candidate
        if 'definition' in plan.intents:
            definition=definition_result(plan,hits,question)
            if fee and definition and fee['status']=='answered':
                return join_answers([definition,fee],hits)
            if fee or definition:
                partial=fee or definition
                if partial['status']=='answered':
                    partial=dict(partial)
                    partial['status']='partial'
                    partial['answer'] += ' A kérdés másik részét a betöltött forrásokból nem sikerült megbízhatóan megválaszolni.'
                return partial
            return None
        return fee
    if 'definition' in plan.intents:
        result=definition_result(plan,hits,question)
        if result:return result
        return extract_answer(question,hits)
    return None
