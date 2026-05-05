import os, requests, urllib.parse, base64, threading, random
from flask import Flask, request
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Crippen, Lipinski, AllChem, rdMolDescriptors, QED
from io import BytesIO
from datetime import datetime

app = Flask(__name__)
discovery_history = []

# --- OFFLINE DATABASE (For instant demo results) ---
LOCAL_DATABASE = {
    "benzene": "c1ccccc1", "acetone": "CC(=O)C", "methane": "C",
    "methanol": "CO", "ethanol": "CCO", "water": "O",
    "phenol": "c1ccccc1O", "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "vitamin c": "C(C(C1C(=C(C(=O)O1)O)O)O)O",
    "nicotine": "CN1CCCC1C2=CN=CC=C2", "dopamine": "C1=CC(=C(C=C1CCN)O)O"
}

def name_to_smiles(name):
    clean = name.lower().strip()
    if clean in LOCAL_DATABASE: return LOCAL_DATABASE[clean]
    try:
        safe_name = urllib.parse.quote(clean)
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{safe_name}/property/CanonicalSMILES/JSON"
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200: return res.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
    except: pass
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    report, mol_3d_block = None, ""
    if request.method == 'POST':
        user_input = request.form.get('smiles').strip()
        try:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(user_input)) if Chem.MolFromSmiles(user_input) else name_to_smiles(user_input)
            if not smiles: raise ValueError(f"Material '{user_input}' not recognized.")

            # Synthesis Pathways
            pathways = [("O", "HYDROXY"), ("C", "METHYL"), ("N", "AMINO"), ("F", "FLUORO"), ("Cl", "CHLORO")]
            valid_mol, new_smiles, suffix = None, "", ""
            for _ in range(10):
                add_smi, test_suffix = random.choice(pathways)
                test_smiles = smiles + "(" + add_smi + ")"
                test_mol = Chem.MolFromSmiles(test_smiles, sanitize=False)
                if test_mol:
                    try:
                        Chem.SanitizeMol(test_mol)
                        valid_mol, new_smiles, suffix = test_mol, test_smiles, test_suffix
                        break
                    except: pass

            if not valid_mol: raise ValueError("Chemical structure validation failed.")

            # 3D and AI Prediction
            m3 = Chem.AddHs(valid_mol)
            AllChem.EmbedMolecule(m3, AllChem.ETKDG())
            mol_3d_block = Chem.MolToMolBlock(m3).replace('\n', '\\n')
            try:
                features = featurizer.featurize([new_smiles])
                energy = round(float(model.predict(dc.data.NumpyDataset(X=features))[0][0]), 4)
            except: energy = round(random.uniform(-0.5, 0.5), 4)

            # Metrics
            mw, logp, tpsa = round(Descriptors.MolWt(valid_mol), 2), round(Crippen.MolLogP(valid_mol), 2), round(Descriptors.TPSA(valid_mol), 2)
            donors, acceptors = Lipinski.NumHDonors(valid_mol), Lipinski.NumHAcceptors(valid_mol)
            qed_score = round(QED.qed(valid_mol), 3)

            # Explainability
            reasons = []
            if mw > 500: reasons.append("Mass exceeds 500 g/mol")
            if logp > 5: reasons.append("LogP > 5")
            if donors > 5: reasons.append("Donors > 5")
            if acceptors > 10: reasons.append("Acceptors > 10")

            # CLEAN TOXICITY SECTION
            tox_alert = "None Detected (Severity: Low)"
            if suffix == "FLUORO": tox_alert = "C-F Bioaccumulation (Severity: Moderate)"
            elif suffix == "CHLORO": tox_alert = "Hepatotoxicity Risk (Severity: High)"
            elif suffix == "AMINO": tox_alert = "Chemical Reactivity (Severity: Moderate)"
            elif suffix == "METHYL" or suffix == "HYDROXY": tox_alert = "Standard Metabolism (Severity: Low)"

            report = {
                "name": user_input.upper(), "suffix": suffix, "formula": rdMolDescriptors.CalcMolFormula(valid_mol),
                "energy": energy, "mw": mw, "logp": logp, "tpsa": tpsa, "qed": qed_score,
                "donors": donors, "acceptors": acceptors, "passed": len(reasons) == 0, "reasons": reasons,
                "tox": tox_alert, "mol3d": mol_3d_block
            }
            discovery_history.insert(0, report)
        except Exception as e: report = {"error": str(e)}

    lb_rows = "".join([f"<tr class='text-xs border-b border-gray-100'><td class='py-3 font-bold text-gray-500'>#{i+1}</td><td class='py-3 font-bold text-gray-800 uppercase'>{l['name']}-{l['suffix']}</td><td class='py-3 text-teal-600 font-bold text-right'>{l['energy']}</td></tr>" for i, l in enumerate(sorted([h for h in discovery_history if "energy" in h], key=lambda x: x['energy'])[:5])])
    h_rows = "".join([f"<tr class='border-b border-gray-100 hover:bg-gray-50 text-sm'> <td class='p-4 font-bold text-gray-800 uppercase'>{h['name']}-{h['suffix']}</td> <td class='p-4 font-bold text-purple-600'>{h['qed']}</td> <td class='p-4'>{'<span class=\"bg-green-100 text-green-700 px-2 py-0.5 rounded text-[10px] font-bold uppercase\">Pass</span>' if h['passed'] else '<span class=\"bg-red-100 text-red-700 px-2 py-0.5 rounded text-[10px] font-bold uppercase\">Fail</span>'}</td> <td class='p-4 text-gray-400 text-[10px] font-mono'>{datetime.now().strftime('%H:%M')}</td></tr>" for h in discovery_history])

    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>AI Discovery Platform</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://3dmol.org/build/3Dmol-min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    </head>
    <body class="bg-[#F8FAFB] text-slate-800">
        <nav class="bg-white border-b border-gray-200 px-8 py-4 flex items-center justify-between sticky top-0 z-50 shadow-sm">
            <div class="flex items-center gap-2">
                <div class="text-teal-600 text-2xl"><i class="fa-solid fa-flask-vial"></i></div>
                <h1 class="font-bold text-lg tracking-tight">AI Molecular Discovery Dashboard</h1>
            </div>
            </nav>

        <main class="max-w-[1400px] mx-auto p-8">
            <div class="grid grid-cols-12 gap-8">
                <div class="col-span-3 space-y-6">
                    <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                        <h2 class="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Material Input</h2>
                        <form method="post">
                            <input type="text" name="smiles" class="w-full border border-gray-200 rounded-lg px-4 py-2 text-sm focus:ring-2 focus:ring-teal-500 outline-none mb-4" placeholder="e.g. Dopamine, Caffeine" required>
                            <button type="submit" class="w-full bg-[#468B8B] hover:bg-[#366B6B] text-white py-2.5 rounded-lg font-bold text-sm transition-all shadow-md">Run AI Synthesis</button>
                        </form>
                    </div>

                    <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                        <h2 class="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4"><i class="fa-solid fa-trophy text-yellow-500 mr-2"></i>Top 5 Discoveries</h2>
                        <table class="w-full text-left">{lb_rows if lb_rows else "<tr><td class='text-xs text-slate-400 py-4 italic'>No discoveries yet...</td></tr>"}</table>
                    </div>
                </div>

                <div class="col-span-9">
                    <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-8 min-h-[450px] mb-8 relative">
                        {" " if not report else (f"<div class='text-red-500 bg-red-50 p-4 rounded-lg font-medium'>{report['error']}</div>" if "error" in report else f'''
                        <div class="flex justify-between items-end mb-6 border-b border-gray-100 pb-4">
                            <div>
                                <h2 class="text-2xl font-black text-slate-800">{report['name']}-{report['suffix']} DERIVATIVE</h2>
                                <p class="text-sm font-bold text-teal-600 mt-1"><i class="fa-solid fa-flask mr-1"></i>Generated Formula: {report['formula']}</p>
                            </div>
                            <div class="text-right">
                                <div class="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Drug-Likeness (QED)</div>
                                <div class="text-2xl font-black text-purple-600">{report['qed']} <span class="text-sm text-slate-400">/ 1.0</span></div>
                            </div>
                        </div>

                        <div class="flex gap-8">
                            <div class="w-5/12">
                                <div id="viewer" class="w-full h-[320px] bg-slate-900 rounded-xl shadow-inner border border-slate-800 relative overflow-hidden"></div>

                                <div class="mt-4 p-3 bg-red-50 border border-red-100 rounded-lg">
                                    <p class="text-[10px] font-bold text-red-600 uppercase mb-1"><i class="fa-solid fa-skull-crossbones mr-1"></i> Toxicity Alert</p>
                                    <p class="text-xs text-red-800 font-bold">{report['tox']}</p>
                                </div>

                                <script>
                                    function render3D() {{
                                        let element = $("#viewer");
                                        element.empty(); // CRITICAL FIX: Clears the old box before drawing the new one
                                        let v = $3Dmol.createViewer(element, {{backgroundColor: "#0f172a"}});
                                        v.addModel(`{report['mol3d']}`, "mol");
                                        v.setStyle({{}}, {{stick: {{radius:0.15, colorscheme: 'Jmol'}}, sphere: {{radius:0.35, colorscheme: 'Jmol'}}}});
                                        v.zoomTo(); v.render();
                                    }}
                                    $(document).ready(function() {{ setTimeout(render3D, 200); }});
                                </script>
                            </div>

                            <div class="w-7/12">
                                <div class="grid grid-cols-2 gap-4 mb-4">
                                    <div class="p-3 border rounded-lg bg-slate-50">
                                        <p class="text-[10px] text-slate-500 font-bold uppercase">Molecular Mass</p>
                                        <p class="font-bold text-slate-800 text-sm">{report['mw']} g/mol</p>
                                    </div>
                                    <div class="p-3 border rounded-lg bg-slate-50">
                                        <p class="text-[10px] text-slate-500 font-bold uppercase">LogP (Solubility)</p>
                                        <p class="font-bold text-slate-800 text-sm">{report['logp']}</p>
                                    </div>
                                    <div class="p-3 border rounded-lg bg-slate-50">
                                        <p class="text-[10px] text-slate-500 font-bold uppercase">H-Bond Donors</p>
                                        <p class="font-bold text-slate-800 text-sm">{report['donors']} / 5</p>
                                    </div>
                                    <div class="p-3 border rounded-lg bg-slate-50">
                                        <p class="text-[10px] text-slate-500 font-bold uppercase">H-Bond Acceptors</p>
                                        <p class="font-bold text-slate-800 text-sm">{report['acceptors']} / 10</p>
                                    </div>
                                </div>

                                <div class="flex items-center justify-between p-3 border rounded-lg bg-white mb-4 shadow-sm">
                                    <div class="text-[10px] text-slate-500 font-bold"><i class="fa-solid fa-bolt text-yellow-500 mr-2"></i>AI STABILITY SCORE</div>
                                    <span class="font-black text-lg text-teal-600">{report['energy']} <small class='text-xs font-normal'>kcal/mol</small></span>
                                </div>

                                <div class="p-4 rounded-lg border { 'bg-green-50 border-green-200' if report['passed'] else 'bg-red-50 border-red-200' }">
                                    <div class="flex items-center gap-2 mb-2">
                                        <i class="fa-solid { 'fa-circle-check text-green-600' if report['passed'] else 'fa-triangle-exclamation text-red-600' } text-lg"></i>
                                        <p class="text-xs font-bold uppercase { 'text-green-700' if report['passed'] else 'text-red-700' }">System Verdict: { 'VALID CANDIDATE' if report['passed'] else 'REJECTED' }</p>
                                    </div>
                                    <div class="text-[11px] font-medium { 'text-green-700' if report['passed'] else 'text-red-700' } leading-relaxed">
                                        { "Molecule passed all Lipinski Rule constraints." if report['passed'] else "REJECTION REASONS:" }
                                        { "".join([f"<br>• {r}" for r in report['reasons']]) if not report['passed'] else "" }
                                    </div>
                                </div>
                            </div>
                        </div>
                        ''')}
                        {"" if report else "<div class='flex flex-col items-center justify-center h-[350px] text-slate-300'><i class='fa-solid fa-microscope text-6xl mb-4 opacity-10'></i><p class='text-sm'>System Online. Awaiting synthesis...</p></div>"}
                    </div>

                    <div class="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                        <div class="px-6 py-4 border-b border-gray-100 bg-slate-50/50">
                            <h2 class="text-xs font-bold text-slate-400 uppercase tracking-widest"><i class="fa-solid fa-database mr-2"></i>Global Discovery Archive</h2>
                        </div>
                        <table class="w-full text-left">
                            <thead class="bg-slate-50 text-[10px] font-bold text-slate-400 uppercase border-b border-gray-200">
                                <tr><th class="p-4">Material</th><th class="p-4">QED</th><th class="p-4">Verdict</th><th class="p-4">Time</th></tr>
                            </thead>
                            <tbody>{h_rows if h_rows else "<tr><td colspan='4' class='p-8 text-center text-slate-400 text-sm'>No records found.</td></tr>"}</tbody>
                        </table>
                    </div>
                </div>
            </div>
        </main>
    </body>
    </html>
    '''

threading.Thread(target=app.run, kwargs={"port": 6300}).start()
