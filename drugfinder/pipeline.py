import requests
import json
import csv
import os
import argparse
from pathlib import Path
from time import sleep
from datetime import datetime
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski
import random
from novabrain_config import OUTPUT_JSONL, OUTPUT_CSV
from novabrain_rag import build_index


# ---------- SAVE HELPERS ----------
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec) + "\n")

def save_csv(data, path):
    if not data:
        return
    keys = sorted({k for rec in data for k in rec.keys() if not isinstance(rec[k], (list, dict))})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k: v for k, v in rec.items() if not isinstance(v, (list, dict))} for rec in data])

# ---------- FDA ----------
def fetch_fda_drugs(limit=100):
    url = "https://api.fda.gov/drug/drugsfda.json"
    skip = 0
    results = []
    while True:
        params = {"limit": limit, "skip": skip}
        r = requests.get(url, params=params)
        if r.status_code != 200:
            break
        batch = r.json().get("results", [])
        if not batch:
            break
        results.extend(batch)
        skip += limit
        print(f"Fetched {len(results)} FDA records...")
        sleep(0.2)
    return results

# ---------- PubChem ----------
def fetch_pubchem_data(name):
    cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
    r = requests.get(cid_url)
    if r.status_code != 200:
        return {}
    cids = r.json().get("IdentifierList", {}).get("CID", [])
    if not cids:
        return {}
    cid = cids[0]
    prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName,CanonicalSMILES,InChI,InChIKey,MolecularWeight/json"
    r2 = requests.get(prop_url)
    if r2.status_code != 200:
        return {}
    props = r2.json().get("PropertyTable", {}).get("Properties", [{}])[0]
    return {
        "pubchem_cid": cid,
        "iupac_name": props.get("IUPACName"),
        "smiles": props.get("CanonicalSMILES"),
        "inchi": props.get("InChI"),
        "inchikey": props.get("InChIKey"),
        "molecular_weight": props.get("MolecularWeight")
    }

# ---------- ChEMBL ----------
def fetch_chembl_data(name):
    search_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={name}"
    r = requests.get(search_url)
    if r.status_code != 200:
        return {}
    mols = r.json().get("molecules", [])
    if not mols:
        return {}
    chembl_id = mols[0]["molecule_chembl_id"]

    mech_url = f"https://www.ebi.ac.uk/chembl/api/data/mechanism.json?molecule_chembl_id={chembl_id}"
    r2 = requests.get(mech_url)
    targets = []
    if r2.status_code == 200:
        for mech in r2.json().get("mechanisms", []):
            targets.append({
                "target_name": mech.get("target_pref_name"),
                "action_type": mech.get("action_type"),
                "mechanism_of_action": mech.get("mechanism_of_action")
            })

    phase_url = f"https://www.ebi.ac.uk/chembl/api/data/drug.json?molecule_chembl_id={chembl_id}"
    r3 = requests.get(phase_url)
    phase, withdrawn = None, False
    if r3.status_code == 200:
        drugs = r3.json().get("drugs", [])
        if drugs:
            phase = drugs[0].get("max_phase")
            withdrawn = drugs[0].get("withdrawn_flag", False)
    return {"chembl_id": chembl_id, "targets": targets, "max_phase": phase, "withdrawn": withdrawn}

# ---------- ClinicalTrials.gov ----------
def fetch_clinical_trials(name):
    base = "https://clinicaltrials.gov/api/query/full_studies"
    params = {"expr": name, "min_rnk": 1, "max_rnk": 5, "fmt": "json"}
    r = requests.get(base, params=params)
    if r.status_code != 200:
        return []
    studies = []
    for s in r.json().get("FullStudiesResponse", {}).get("FullStudies", []):
        sid = s["Study"]["ProtocolSection"]["IdentificationModule"]["NCTId"]
        phase = s["Study"]["ProtocolSection"]["DesignModule"].get("PhaseList", {}).get("Phase", [])
        status = s["Study"]["ProtocolSection"]["StatusModule"]["OverallStatus"]
        studies.append({"nct_id": sid, "phase": phase, "status": status})
    return studies

# ---------- USPTO ----------
def fetch_patents_and_status(name):
    url = "https://api.patentsview.org/patents/query"
    query = {
        "q": {"_text_any": {"patent_abstract": name}},
        "f": ["patent_number", "patent_date"],
        "o": {"per_page": 1}
    }
    r = requests.post(url, json=query)
    if r.status_code != 200:
        return "No Patent Found"
    patents = r.json().get("patents", [])
    if not patents:
        return "No Patent Found"
    patent_date = patents[0].get("patent_date")
    try:
        year = int(patent_date.split("-")[0])
        if datetime.now().year - year > 20:
            return "Expired"
        else:
            return "Active"
    except:
        return "Unknown"

# ---------- Similarity ----------
def calc_similarity(smiles_a, smiles_b):
    if not smiles_a or not smiles_b:
        return None
    try:
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2)
        return round(DataStructs.TanimotoSimilarity(fp_a, fp_b), 3)
    except:
        return None

# ---------- Drug-likeness ----------
def drug_likeness_score(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return 0
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        if 100 < mw < 700 and -1 < logp < 6 and hbd <= 5 and hba <= 10:
            return 1.0
        return 0.5 if mw < 900 else 0.0
    except:
        return 0

# ---------- RDKit Mutation ----------
def mutate_molecule(smiles, num_mutations=3):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []
    analogs = []
    for _ in range(num_mutations):
        editable = Chem.RWMol(mol)
        mutation_type = random.choice(["add_atom", "remove_atom", "change_bond"])
        if mutation_type == "add_atom":
            atom_idx = random.randint(0, editable.GetNumAtoms() - 1)
            new_atom = Chem.Atom(random.choice([6, 7, 8, 9]))  # C, N, O, F
            editable.AddAtom(new_atom)
            editable.AddBond(atom_idx, editable.GetNumAtoms() - 1, Chem.BondType.SINGLE)
        elif mutation_type == "remove_atom" and editable.GetNumAtoms() > 5:
            idx_to_remove = random.randint(0, editable.GetNumAtoms() - 1)
            editable.RemoveAtom(idx_to_remove)
        elif mutation_type == "change_bond" and editable.GetNumBonds() > 0:
            bond_idx = random.randint(0, editable.GetNumBonds() - 1)
            bond = editable.GetBondWithIdx(bond_idx)
            current_type = bond.GetBondType()
            new_type = Chem.BondType.DOUBLE if current_type == Chem.BondType.SINGLE else Chem.BondType.SINGLE
            bond.SetBondType(new_type)
        new_mol = editable.GetMol()
        Chem.SanitizeMol(new_mol)
        analog_smiles = Chem.MolToSmiles(new_mol)
        mw = Descriptors.MolWt(new_mol)
        logp = Descriptors.MolLogP(new_mol)
        if 100 < mw < 700 and -1 < logp < 6:
            analogs.append(analog_smiles)
    return list(set(analogs))

# ---------- MAIN ----------
def main():
    fda_records = fetch_fda_drugs(limit=100)
    dataset = []
    marketed_smiles = []

    for rec in fda_records:
        openfda = rec.get("openfda", {})
        base_name = (openfda.get("generic_name") or openfda.get("brand_name") or [""])[0]
        chem = fetch_pubchem_data(base_name)
        if chem.get("smiles"):
            marketed_smiles.append(chem["smiles"])

    for i, rec in enumerate(fda_records, 1):
        openfda = rec.get("openfda", {})
        products = rec.get("products", [])
        base_name = (openfda.get("generic_name") or openfda.get("brand_name") or [""])[0]
        if not base_name:
            continue

        print(f"[{i}/{len(fda_records)}] {base_name}...")
        entry = {
            "drug_name": base_name,
            "brand_names": openfda.get("brand_name", []),
            "generic_names": openfda.get("generic_name", []),
            "manufacturer_name": openfda.get("manufacturer_name", []),
            "application_number": rec.get("application_number"),
            "marketing_status": products[0].get("marketing_status") if products else None
        }
        entry.update(fetch_pubchem_data(base_name))
        entry.update(fetch_chembl_data(base_name))
        entry["clinical_trials"] = fetch_clinical_trials(base_name)
        entry["patent_status"] = fetch_patents_and_status(base_name)

        if entry.get("smiles"):
            sims = [calc_similarity(entry["smiles"], s) for s in marketed_smiles if s]
            sims = [s for s in sims if s is not None]
            max_sim = max(sims) if sims else 0
            entry["similarity_to_marketed"] = max_sim
            entry["novelty_score"] = round(1 - max_sim, 3)
            entry["drug_likeness_score"] = drug_likeness_score(entry["smiles"])
        else:
            entry["novelty_score"] = 0
            entry["drug_likeness_score"] = 0

        if entry["patent_status"] in ["Expired", "No Patent Found"]:
            patent_weight = 1.0
        elif entry["patent_status"] == "Active":
            patent_weight = 0.2
        else:
            patent_weight = 0.5
        entry["patent_opportunity_score"] = round(entry["novelty_score"] * patent_weight, 3)

        if entry["marketing_status"] and "approved" in entry["marketing_status"].lower():
            rd_class = "Approved"
            rd_score = 0.2
        elif entry.get("max_phase") and entry["max_phase"] > 0:
            rd_class = "Investigational"
            rd_score = 0.7
        elif entry.get("withdrawn") is True:
            rd_class = "Abandoned"
            rd_score = 0.9
        else:
            rd_class = "Experimental"
            rd_score = 0.8
        entry["rd_class"] = rd_class
        entry["rd_attractiveness"] = round((entry["novelty_score"] + entry["patent_opportunity_score"] + rd_score) / 3, 3)

        if entry["patent_opportunity_score"] > 0.7 and entry["rd_attractiveness"] > 0.7:
            entry["designation"] = "Hot Patent Opportunity"
        elif entry["rd_class"] == "Abandoned" and entry["drug_likeness_score"] > 0.7:
            entry["designation"] = "Repurposing Candidate"
        elif entry["novelty_score"] > 0.85:
            entry["designation"] = "Breakthrough Potential"
        else:
            entry["designation"] = "Low Value"

        if entry["patent_status"] in ["Expired", "No Patent Found"] and entry.get("smiles"):
            analogs = mutate_molecule(entry["smiles"], num_mutations=5)
            entry["suggested_novel_analogs"] = []
            for a in analogs:
                status = fetch_patents_and_status(a)
                sims = [calc_similarity(a, s) for s in marketed_smiles if s]
                max_sim = max(sims) if sims else 0
                novelty = round(1 - max_sim, 3)
                drug_like = drug_likeness_score(a)
                pat_score = round(novelty * (1.0 if status in ["Expired", "No Patent Found"] else 0.2), 3)
                entry["suggested_novel_analogs"].append({
                    "smiles": a,
                    "patent_status": status,
                    "novelty_score": novelty,
                    "drug_likeness_score": drug_like,
                    "patent_opportunity_score": pat_score
                })

        dataset.append(entry)
        sleep(0.2)

    save_jsonl(dataset, OUTPUT_JSONL)
    save_csv(dataset, OUTPUT_CSV)
    print(f"✅ Saved {len(dataset)} compounds to {OUTPUT_JSONL} and {OUTPUT_CSV}")

    if OUTPUT_JSONL.exists():
        try:
            from novabrain_rag import build_index
            print("\n🔄 Building NovaBrain vector index from latest dataset...")
            build_index()
            print("✅ NovaBrain vector index updated.")
        except Exception as e:
            print(f"❌ Failed to update NovaBrain vector index: {e}")
    else:
        print("❌ NovaBrain vector index not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NovaBrain data pipeline")
    parser.add_argument("--dataset", type=str, help="Base name for dataset files (no extension)")
    parser.add_argument("--backend", type=str, choices=["faiss", "pinecone", "weaviate"], help="Vector storage backend")
    args = parser.parse_args()

    # Override config from command line
    if args.dataset:
        os.environ["NOVABRAIN_DATASET"] = args.dataset
    if args.backend:
        os.environ["NOVABRAIN_VECTOR_BACKEND"] = args.backend

    # Reload config after overrides
    from importlib import reload
    import novabrain_config
    reload(novabrain_config)

    main()
    print("🔄 Building vector index...")
    build_index()
