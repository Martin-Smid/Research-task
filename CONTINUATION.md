# Pokračování práce — working_with_chat

## START HERE (2026-11-02): mapa dokumentů, stav, co dál
Jak se v tom vyznat (nové Claude sezení čte v tomto pořadí):
1. `CLAUDE.md`: pravidla práce (krátké). Seznam "Known issues" (body 1-16), session logy a handoff 2026-10-27 jsou na KONCI tohoto souboru ("Archiv"). 2. Tento soubor: pracovní deník (česky); starší sekce níže jsou HISTORIE, platí pořadí podle dat.
3. `plans.md` (git-ignored schválně, repozitář je veřejný; existuje jen na Koios, zálohovat; anglicky): výzkumný plán a všechna čísla; **sekce 11 = závěry, kam co vede, otevřené úkoly, rozhodnutí, index souborů**; sekce 7-10 = chronologický záznam.
4. `tests/README.md` (testy a měřicí skripty), `tests/spin/README.md` (spin), `docs/` (`crosscode_comparison/` = jaxion a PyUltraLight, `t5/`, `paper1_outline.tex` = osnova článku k vložení do Overleafu, `clanek_BC_obsah.md` = přehled článku BC).

Stav k 2026-11-02 (větev `working_with_chat`, Koios, poslední push uživatele byl `203b21d`):
- Hotové a ověřené: sink-sink gravitace, slučování, dynamické tření vůči přesné lineární teorii (T1-T3), `smooth_force`, soliton s BH (1D/3D), `correct_params.py`, `strict_time_step`, T5 (reprodukce t_stop z arXiv:2403.09038: 45,9 vs 46,9 Myr, jeden běh), srovnání se dvěma dalšími kódy (jaxion: shoda 0,2-2,6 %, PyUltraLight: shoda jen pro M_Q >= 1), spin testy (5 testů + měření skalár vs spin 1, 2).
- Všechny testy na P100 prošly 2026-11-02 (job 3650913: `run_all`, 5 spin testů, `compare` a `diagnostics-check` bitově shodné). Spouští je `sbatch slurm/koios_tests_all.sbatch` (~10 min).
- Není hotové / nepotvrzeno: dlouhý stacionární běh BH v solitonu (15-30 Myr; pravidlo r_B >= 4-6 buněk není potvrzené), příčina driftu E (0,6-0,8 %) a hybnosti v T5, T6 (citlivost na eps/dx, N, okraje), nezávislé T5 v jiném kódu, limita eps -> 0 pro M_Q <= 1 nemá přesnou referenci, literární kontrola novosti.
- Rozhodnutí na uživateli: výchozí `smooth_force` v kódu (zatím False, doporučeno True pro BH v solitonu), spin testy do `run_all`, (rozhodnuto 2026-11-02: `plans.md` zůstává mimo git, repo je veřejné; CONTINUATION.md, CLAUDE.md a docs/ veřejné jsou).
- Data: 2026-11-02 smazány snapshoty a checkpointy sedmi velkých běhů v `resources/data/` (99 GB -> 218 MB), zůstaly `run_config.json`, CSV a grafy; jak je znovu spustit je v `plans.md` 11.8 (T5: `sbatch --export=ALL,T_MYR=60 slurm/koios_t5.sbatch`, ~8 min).
- Pravidla, která se nemění: commit jen na vyžádání a na jméno uživatele (Martin-Smid, bez Co-Authored-By), push nikdy; mazat jen podle jmen, která uživatel schválí; novost nic nepřipisovat, dokud není přečtena literatura; čísla z agentů (jaxion, PyUltraLight) jsou nezkontrolovaná a srovnání není slepé.
- Provozní: připojení `ssh koios` (Windows ssh-agent: `Start-Service ssh-agent`, potom `ssh-add`), skripty psát do souboru a `scp` (PowerShell kazí uvozovky), `set -u` v sbatch s EESSI zabije job, jedna volná GPU, N=384 nejde na P100, `/tmp` je lokální na uzlu, `--export=…MQ_LIST=1,0.3` rozdělí hodnotu čárkou. Další pasti v `plans.md` 11.7.

## TL;DR (2026-11-06)
Hotovo, podrobnosti v `CLAUDE.md` (Known issues a session logy):
- **Běh a reference:** Koios P100, `baselines_koios_p100/` bitově reprodukovatelná (`KOIOS.md`). Quadro vs P100: baryony 1e-14 %, vlna ~1e-4 %, sink ~1-3 %.
- **Plyn** (issue 1, log 2026-10-16): CFL přepočet před každým substepem, vakuová buňka mezi toky, energie gravitace přes stěny buněk (kick-hydro-kick, pojistka vnitřní energie), validace Sod, Jeans, volný pád. Stabilitu plynu bereme jako hotovou. Zbývá numerická viskozita (jedna Eulerova fáze MUSCL): ztráta momentu hybnosti −30,9 / −18,2 / −9,9 % pro N = 64 / 128 / 256. Plyn umí jen řád 2 (`Evolution_Class.py:78-79` jinak vyhodí `ValueError`).
- **ψ v `complex128`** (issue 7) a konvergence soliton + baryony N = 64/128/256 (issue 10): drift energie −0,036 / −0,024 / −0,015 %, hmotnost ψ ~1e-10 %.
- **Sinky** (issues 5, 11-15): potenciál a energie akrece, sink-sink gravitace, slučování, dynamické tření, `smooth_force`, `strict_time_step`, tvorba sinku (sekce 2026-11-05 níže), lokální Bondi.
- **Testy** `tests/` (gas / particle / sink / other, spin zvlášť), `run_all.py` ~1 min. Restart (`resume`, `start_new_segment`) bitově shodný. Dvoufázový setup galaxie (relaxace, pak `from_checkpoint` + `start_new_segment`): `examples/variety/main.py`.
- **Dokumentace:** README, Sphinx v `docs/` (installation, user_guide, simulation_class, api, rotation_curves).

Otevřené (ověřeno v kódu 2026-11-06):
1. **Částice a sinky: opravitelné 2026-11-06, ale výchozí je vypnuto** (`Simulation_Class.baryon_time_centered_force`, výchozí False, ~1,7× dražší krok, jen řád 2; `_time_centered_forces` v `Evolution_Class.py`): síla je průměr ze začátku a ze zkušebního konce kroku, test dýchajícího solitonu 1,3e-3 → 2,0e-5 r0, řád 2,21. Proti P100 baselines se výsledky mění nejvýš o 5e-6 (relativně); v realistickém případu (soliton + hvězdy + sink) o ~0,1 % a chyba vůči jemnému kroku klesne jen o ~16 %, převládá jiná chyba (příčina neměřena, hypotéza: vazba ψ a částic nebo podkroky). Řády 4/6 pořád zamrzlá síla.
2. **Bondi akrece:** lokální `cs` a rychlost plynu vůči sinku existují (`sink.bondi_local = True`, `Sink_N_Body.py:16-17`, `:488`), ale **výchozí je False**, tedy konstantní `gas.cs`. Rychlost akrece bereme jako numerický test, ne fyzikální predikci.
3. **Samointerakce** (issue 3): `Propagator_Class.get_self_int_potential` pořád počítá `pref*(psi^2 + 2 rho psi)`, `self_int` má výchozí True. Rozhodnout: opravit na g|psi|^2 s testem, nebo vypnout a zdokumentovat.
4. **Výchozí hodnoty k rozhodnutí:** `smooth_force` (False), `strict_time_step` (False; při `h` nad limitem jen varování a běh je tichý nesmysl).
5. **Reference Quadro / RTX / GTX** jsou zastaralé (`complex128`, sinky). Uživatel 2026-11-06: přegenerovat asi není potřeba, rozhodnout před vydáním.
6. **Akrece ULDM do sinku** a plán dalších kroků: sekce 2026-11-05 níže.
7. **Vydání:** viz release checklist níže.

## Cíl a pravidla
Publikovatelný framework: správnost a reprodukovatelnost před rychlostí vývoje. Malé, čitelné změny; po každé změně numeriky porovnat mřížky, částice a energie s referencemi, změna výsledků = zastavit a vyšetřit. Referenční data nepřepisovat jako opravu testu. `working_with_chat` nese testy, reference a historii; `main` jen ověřený kód. Tento soubor patří jen do pracovní větve.

Commity jsou pouze na jméno autora (Martin-Smid), bez `Co-Authored-By` a jiných řádků s Claude/Anthropic (platí i pro Claude Code na Koios).

## Nástroje pro soliton s černou dírou (2026-10-29; necommitnuto vše, co není v `git ls-files`)
- `DressedSoliton.py` (1D, CPU/scipy): radiální základní stav SP s centrální BH (Gaussovskou o šířce `eps` = `softening_bh`, nebo bodovou), shooting v jednom vlastním čísle. `python DressedSoliton.py --selftest`; `--m 1e-21 --M_sol 5.57e7 --M_bh 4e6 --eps 0.005 --out profil.dat` zapíše profil (`r phi U`) pro `Wave_vector_class(packet_type=profil.dat, desired_soliton_mass=vypsaná hmotnost)`. Z Python: `solve_physical(m_ev, M_bh, M_sol, eps)`.
- `DressedSoliton3D.py` (3D, GPU): imaginary-time Strang na mřížce simulace (stejná periodická Poissonova rovnice, analytický Gaussovský potenciál sinku), `relax(sim, m_sol, m_bh, bh_position, eps, pin_centroid=None)`, `save`, `load_into(wave_function, soubor.npz)` po `initialize_simulation()`. `pin_centroid` drží soliton v bodě (vázaný stav deformovaný slapem BH, ne stacionární). CLI: `python DressedSoliton3D.py --N 192 --half_box 1.0 --M_bh 4e6 --eps 0.005 --out dressed.npz`.
- `correct_params.py`: z N, M_sol, m, M_bh vyplní box, ε, H, čas a kontroly (pravidla r_B ≥ 4 buňky, ε = 1 buňka, fáze potenciálu < 0,1 jsou odhad).
- Ověření: 3D (BH ve středu, N = 128, box 1,5 kpc) vs 1D: centrální hustota 1,1575e11 vs 1,1584e11 (0,08 %), reziduum 1e-4; vodíkový limit hmotnosti přesně (1e-8). Test `tests/other/test_dressed_soliton.py` (mimo `run_all`, ~5 min na GPU).
- Neplatí: profily z `ModosFinder.py` / `ModosDbscan*.ipynb` mají φ(0) = 1 (pevná centrální hustota), přeškálování hmotností je s BH neplatné, hledání je náhodné.
- Pasti: `wave_function.psi` je do `evolve()` numpy; loader profilů čte float32, interpoluje lineárně a mimo tabulku dává 0 (proto první řádek profilu r = 0); `ModosFinder` má G o 3e-5 vyšší než astropy.

## Článek A: otestovaný kód pro ostatní (software + validace)
Cíl: SP s Fourierovým split-step, baryony (částice, plyn) a sinky; srovnatelné s PyUltraLight, s testy a návodem. Možná místa: JOSS, Astronomy & Computing, RNAAS. Poctivě: software + validace, ne nový fyzikální výsledek.
Hotovo: regrese bitově na P100, testy (Kepler, binárka sinků, slučování, `smooth_force`, Sod, Jeans, volný pád, tvorba sinku, Boss & Bodenheimer), exaktní lineární benchmark v boxu (`tests/sink/sink_friction.py`, `sink_friction_scan.py`, `t2_aggregate.py`), `complex128`, solitonové řešiče, srovnání s jaxion a PyUltraLight (`docs/crosscode_comparison/`, skripty psal agent).
Chybí (ověřeno 2026-11-06):
1. Licence (`LICENSE`) a citace (`CITATION.cff`, DOI přes Zenodo): v repu nejsou, řešit s uživatelem.
2. Dokument „known limitations": periodický box (zbytková gravitace obrazů, střední hustota −(4π/3) G m r/V platí do r ≈ L/4), sponge je `complex64` (`Simulation_Class._build_sponge_potential`), bodový / Gaussovský BH, plyn není jádro článku.
3. Popis parametrů sinku a doporučené rozsahy (ε/λ ≲ 0,15, λ ≥ 3 buňky); stránka o numerice v `docs/` není.
4. Výkon a paměť (P100: N = 384 nejde, ~5 min na případ N = 256), tabulka a škálování.
5. Nezávislý kód na stejném testu a T5 (jaxion a PyUltraLight jsou nerecenzované), jednotná mapa platnosti (viz článek BC).

## Článek BC: tření a pád sinku (černé díry) ve vícesložkovém ULDM
Přehled pro všechna sezení: `docs/clanek_BC_obsah.md` (tam jsou i pozadí, soliton a plán). Spojuje původní B (tření sinku, skalár) a C (vícesložková pole pod gravitací). Samostatné C jako „nový zákon n-závislosti" padlo: Jain+ 2304.01985 změřili τ ≈ N τ0 pro rovné zastoupení.
Motivace (spolupracovník Erick, 2026-11-03): kondenzace v kinetickém režimu je v literatuře hotová (Levkov+ 1804.05857, Jain+, Chen+ 2304.01965), zajímavější je tření a pád BH ve vyšším spinu. Gravitace je U(N)-symetrická, takže n-závislost může jít jen přes statistiku (zrnitost ∝ 1/n, interference polarizací), ne novou sílu (očekávání, neověřeno).
Osnova (uživatel): (1) úvod, problém, teorie vyšších spinů; (2) testy M_Q a ε/λ, tři režimy, kde přestává platit lineární teorie, srovnání s jaxion a PyUltraLight; (3) IC: (A) gaussovské pole vyzrálé do solitonu, sink do vzdálenosti r, doba pádu a pohyb, (B) stabilní soliton se sinkem vs bez (volný vs dressed); (4) rozlišení, krok, M_BH.

### Co je změřeno
**Sink v uniformním pozadí (skalár).** M_Q ≥ 3: simulace = přesná lineární teorie pro konečný periodický box do 0,8 %; M_Q ≤ 1 tření osciluje (fyzikální), střední hodnota = ustálený stav při M_Q = 1, o 15 % výš při 0,3. Pravidlo ε ≲ 0,3 λ, λ ≥ 3 buňky, pod 0,25 buňky se teorie nemění (< 0,3 %). Bodový přijímač porušuje hybnost a energii ∝ (ε/λ)², `smooth_force=True` ji obnoví (pod ~2 %, ~N^-2). Konvergence v N a H pod 0,5 %. Limita ε → 0 souhlasí s Lancasterem+ 2020 (C 3,276 vs 3,268 při M_Q 17,4; M_Q = 1 +2 %; M_Q = 0,3 +7-9 %). Chybí přesná reference pro ε → 0 při M_Q ≤ 1. T5 (BH v klidu v solitonu): 45,9 vs 46,9 Myr (jeden běh), ale drift E_total 0,6-0,8 % a rostoucí hybnost ψ + BH (příčina otevřená).

**Čas vzniku solitonu vs počet složek.** IC: gaussovské pole, box 19,2 kpc, m = 2,5e-22 eV, ρ_mean 5e5 Msun/kpc³, řez |k| ≤ 8 módů, N = 64, 3 seedy, jen vlastní gravitace. Variance(ρ/ρ_mean) · n = 1 (testy `test_spin_granule_statistics.py`, `test_spin_independent_fields.py`). Čas zlomu [Gyr] (k = 3 / k = 5; n_eff = 1/Σf², u 1,52 a 2,27 jde o spin 1 s nerovnými zlomky, N = 6, 8, 9 jsou složená N-složková pole, ne spin 2):

| n_eff | 1 | 1,52 | 2,27 | 3 (spin 1) | 5 (spin 2) | 6 | 8 | 9 (spin 4) |
|---|---|---|---|---|---|---|---|---|
| k = 3 | 9,2 ± 1,7 | 11,1 ± 0,8 | 15,6 ± 1,7 | 31,7 ± 5,2 | 48,9 ± 7,8 | 66,1 ± 8,8 | 96,0 ± 6,0 | 98,9 ± 4,6 |
| k = 5 | 9,7 ± 1,4 | 11,5 ± 0,5 | 18,1 ± 2,3 | 39,6 ± 3,2 | 63,2 ± 10,3 | 81,6 ± 5,0 | 113,9 ± 14,0 | 125,9 ± 3,3 |

Fit a·N^p: p = 1,25 ± 0,05 (k = 3), 1,33 ± 0,05 (k = 5); lineární τ ∝ N i exponenciála jsou vyloučeny, ale jen N = 64, 3 seedy, jedna definice zlomu (p závisí na k) a hrubší sampling u N ≥ 6 (0,8 Gyr). Skript `tests/spin/onset_times.py`, graf `resources/data/spin_onset_vs_neff.png`. Vzorec Jain+ t_a/t_skalár = 1/(2 f_a − Σf²) pro nerovné zlomky je **moje čtení rovnice 6, neověřené**; naměřeno je o 15-35 % pomalejší. Chen+ 2304.01965 nečteno.

**Pád sinku do zralého solitonu** (N = 128; fáze 1: skalár 12 Gyr, spin 1 60 Gyr, spin 2 90 Gyr; fáze 2 `tests/spin/sink_infall.py`: sink 2,5e7 Msun, 4 kpc od píku, v klidu, `smooth_force`, softening 1,5 buňky, 3 Gyr; 5 / 5 / 3 seedy skalár / spin 1 / spin 2):
- První pád k solitonu 0,105-0,113 Gyr, nezávisí na spinu, odpovídá volnému pádu v hmotě uvnitř 4 kpc (0,11 Gyr).
- Pozdní usazení (střední vzdálenost sink-pík v 2-3 Gyr) 0,76 / 0,50 / 0,27 kpc, bloudění píku bez sinku 1,25 / 0,58 / 0,47 kpc: usazení sleduje bloudění solitonu, které s n klesá. Sink bloudění nezměnil. Pík je rozlišen na buňku 0,15 kpc, údaje pod ~0,3 kpc nejsou spolehlivé.
- Citlivost (seed 1): pozdní usazení závisí na M_BH (10× menší se neusadí, tlumení ∝ M), softeningu (3 buňky nechá sink ~1 kpc od středu) a na H (bloudění solitonu je chaotické); doba prvního pádu na ničem z toho nezávisí.
- **Je to zatím „bloudění solitonu klesá s n", ne nový efekt tření. O spinu jako příčině nepsat.** Rozdíl skalár vs spin 1 je v rámci změn při změně H a softeningu.

### Plán (pořadí)
1. Zrnitá síla na lehké sinky vs n (`tests/spin/granular_force.py`, běží na pozadí k 2026-11-06, výsledky zatím nejsou): ⟨Δv²⟩(t) a a_rms² pro n = 1, 3, 4, 5, 6 před kolapsem, očekávání a_rms² ∝ 1/n; pak těžký sink v pohybu.
2. Rozlišení: fáze 1 při N = 192/256 (jádro má při N = 128 jen ~1,2 buňky); spin 1 se sinkem menší hmotnosti (blíž lineárnímu tření).
3. Multikomponentní sink: energie a hybnost sinku + ψ, zda `Diagnostics` sčítá všechny složky (zatím jen vizuálně); tlumení přes energii sinku místo jednoho čísla.
4. Zahřívání tracerů a rozptyl síly vs n; přečíst Chen+ 2304.01965.
5. T6 (ε/dx, N, sponge vs periodické okraje), nezávislý kód na T5, adiabatický růst BH, mapa M_Q,eff v solitonu.
6. Vícesložkový stacionární soliton: profil umíme jen pro stejné polarizace, pro různé rozšířit hledání, nebo použít stejné polarizace.
7. Akrece ULDM do sinku (sekce 2026-11-05).
Novost nelze tvrdit před přečtením celých textů Boey+2024 (2403.09038), Wang & Easther 2021, Boey+2025 (2504.16348), Buehler & Desjacques (2207.13740), Lancaster+2020, Jain+ 2304.01985, Chen+ 2304.01965.

## Příprava na vydání (release checklist, stav 2026-11-06)
Hotovo: audit repa (`requirements.txt` smazán, závislosti v `pyproject.toml`; `.idea` není tracked; duplicitní `system_fucntions.py` odstraněn, zůstává `resources/Functions/system_fucntions.py`; volné skripty přesunuty do `examples/variety` a `plots/`), README (instalace), Sphinx dokumentace.
Zbývá (každý bod malá samostatná změna, po každé `tests/run_all.py` a benchmarky, fyzika jen se schválením):
1. Ověřit verze Pythonu a CuPy v `pyproject.toml` (nezkontrolováno).
2. Dokumentace numeriky (FFT split-step, Poisson, sink, plyn) a přehled testů a validací s tolerancemi; formát checkpointu.
3. Samointerakce (otevřený bod 3), výchozí `smooth_force` a `strict_time_step` (bod 4): rozhodne uživatel.
4. Reference sady Quadro / RTX / GTX a `hashes_*.json` (bod 5).
5. `LICENSE`, `CITATION.cff`, DOI; skripty k článkům případně do vlastního repa se závislostí na commitu.
6. CI: v repu není (`.github` chybí); testy potřebují CuPy / GPU, to zdokumentovat.
Co nedělat bez ptaní: commit / push, `main`, `main.py`, `plot_from_snapshots.py`, mazání dat, spouštění na GPU, když běží joby (notebook RTX 3070 šetřit, Koios `gpu3`/`gpu4` jen po dohodě).

## 2026-11-05: tvorba sinku (sink formation), stav a co dál
Hotové (commit 9a4a98c + tento): `Simulation_Class(sink_formation={...})` funguje. Jedno kritérium pro částice, plyn a ULDM (`check_and_create_sinks`, `Sink_N_Body.py`):
- semínko = lokální maximum celkové hustoty (částice + plyn + |psi|^2 pro `include_uldm=True`), nad prahem `consecutive_steps` kontrol; nejhustší první; nic do 2 dx od existujícího sinku; přeskočená buňka se neblokuje navždy;
- sink pohltí VŠECHNY částice v okruhu (`gas_r_acc_cells` = 2 buňky) i přebytek plynu; vznikne v těžišti pohlceného, s jeho hmotou a hybností; kinetická i vnitřní energie plynu jde do `E_diss_formation_total`;
- práh: `density_threshold` (explicitní, vyhrává) > `gas_density_threshold` (číslo, nebo automatický Truelove s konstantním `gas.cs`) > částicový z `baryon_sigma_floor` (teď opravdu podlaha: max(sigma, floor));
- NOVÉ: `dynamic_threshold=True` = Truelove práh přepočítaný při každé kontrole z lokální rychlosti zvuku plynu gamma*P/rho (po buňkách) a aktuální 1D disperze částic (jedna globální hodnota), nebo floor;
- NOVÉ: `federrath_checks=True` = konvergující tok + gravitačně vázaná oblast (Federrath 2010; test minima potenciálu nahrazen podmínkou lokálního maxima hustoty);
- NOVÉ: `sink.bondi_local = True` = Bondi-Hoyle-Lyttleton s lokálním cs a rychlostí plynu vůči sinku (default False = konstantní `gas.cs`, výsledky beze změny).
Testy: `tests/sink/test_sink_formation.py` (a-n), `tests/sink/test_boss_bodenheimer.py` (m = 2 kolaps -> 2 sinky; s Federrathem zůstávají 2 při N = 64 i 128, bez něj 7 / 23), `tests/sink/test_sink_bondi_local.py`; grafy `tests/sink/plot_boss_bodenheimer.py` (`plots/boss_bodenheimer/`). Na P100 (Koios, job 3651759, kopie `~/projects/rt_test`): `run_all` + `compare` + `diagnostics-check` vše OK a bitově shodné.
Pozor: názvy snapshotů mají čas na 6 míst v Gyr (1 kyr), pro krátké běhy ukládat méně často než ~12 kroků nejde; `gas_snapshot_*` se neukládá (jen `baryons_` a `total_density_`).

Co dál (pořadí):
1. **Akrece ULDM do sinku** (rozhodnout s uživatelem). arXiv:2607.00349 (Ludwig, Mocz, Robles) je o akreci PLYNU: fixní poloměr r_sink = 2.5 dx, odebraná hmota jde do BH, cílová rychlost Bondi-Hoyle 4 pi lambda (GM)^2 rho_inf / (cs^2 + v_rel^2)^(3/2), rho_inf z tenké slupky těsně za r_sink; ULDM akreci NEMÁ. Kandidáti na předpis pro ULDM (ověřit, nic neimplementovat z paměti): Unruh 1976 (absorpční průřez skalárního pole, s-vlna), 1905.11745 (příloha, odhad rychlosti pro soliton), 1805.00122, 2309.02482, 1107.3173. Rozhodnout: odstraňovat hmotu z |psi|^2 (mění normu psi, ztráta unitarity), jak.
2. Bondi: rho_inf a v_rel ze slupky vně r_sink (podle 2607.00349), omezení rychlostí dopadu při r_B < dx, dynamický Truelove i pro částice lokálně (disperze po buňkách).
3. B&B: vyšší rozlišení / s akrecí; porovnat s literaturou (vzdálenost binárky ~ stovky AU).
4. Úklid: `docs/simulation_class.md` je zatím necommitnutý (jiná session), diagnostics-check a optimization-check na Quadro nešly (Windows PermissionError při zápisu run_config.json; na P100 prošly).

## Archiv: dřívější obsah CLAUDE.md (přesunuto 2026-11-06; anglicky, beze změn obsahu)
Známé problémy 1-16 s čísly, session logy 2026-10-16 / 10-26, poučení a handoff 2026-10-27. Platná pravidla práce jsou teď v krátkém `CLAUDE.md`. Číslování issues zůstává (odkazy typu "issue 11" platí).

### Known issues — status
1. Gas CFL / FloatingPointError in NBodyGas:
   - **Fixed 2026-10-16:** `dt_sub` was computed once per window; in a collapse the true Courant number grew
     from 0.5 to ~3 and the run ended in NaN. Now `_substeps` re-evaluates it before every substep.
     Emptied cells (below `rho_floor`) also lose their momentum/energy (`_reset_emptied_cells`).
   - **Fixed 2026-10-26 (the window disk case):** independent limiting of rho and momentum left a vacuum cell
     between two diverging flows with finite momentum but floored density, i.e. a face velocity ~1e11; Rusanov
     fluxes of 1e10 made negative densities, whose clamp *created* mass. `_sweep_axis` now falls back to the
     first-order cell states on faces whose velocity exceeds 10x (|v|+c) of the two adjacent cells. Regression
     baselines stay bitwise identical; test `tests/test_gas_vacuum_hole_between_flows.py`. Reproducer:
     `resources/diagnostics/cfl_disk_repro.py`.
   - MUSCL with a single forward-Euler stage is untouched (limits time accuracy of the hydro).
2. Baryon coupling in time: gas now uses a kick–drift–kick with a refreshed potential (second order for the
   gas); particles and sinks (order 2) feel the force averaged over the step (`Simulation_Class.baryon_time_centered_force`,
   opt-in, default False (2026-11-06): trial drift to the end of the step, force rebuilt there, mean of start and end forces; second
   order in h, breathing-soliton error 1.3e-3 -> 2e-5 r0, apparent order 2.21, `tests/particle/test_particle_time_centered_force.py`;
   one extra baryon drift + Poisson solve per step, ~1.7x per step with 20000 stars + a sink; with it on the P100 reference cases
   change by <= 5e-6, a soliton + stars + sink run by ~0.1 %, and the error against a fine step falls only ~16 %; off = frozen
   force, bitwise the old results); orders 4/6 still use the frozen force
   and call particle drift with kick coefficients; gas supports order 2 only.
3. Self-interaction potential `pref*(psi^2 + 2 rho psi)` looks complex/non-unitary; expected `g|psi|^2`. Not touched.
4. Then: cleanup/packaging (requirements.txt is UTF-16 (tqdm added 2026-10-26); `.idea` tracked; duplicate
   `system_fucntions.py`) → docs → settle licence/citation with me.

5. Sinks (fixed 2026-10-26): (a) the analytic sink potential ignored the box's lower boundary, so in a [-L, L] box
   a sink at the centre pulled towards the corner (-L, -L, -L); (b) `accrete_from_gas` removed density but not the
   energy of the cell (specific energy and CFL exploded). Both fixed with tests; accretion is off unless
   `sim.sink_accretion = True` (the call in the main loop was commented out). 5 sink reference files per set
   changed; only `baselines_koios_p100/` was regenerated, the Quadro/RTX/GTX sets are stale for sink cases.
   Older BH runs in boxes that do not start at 0 were affected.
6. Open observations: (the ULDM wave-function mass loss of 0.04-0.09 % per Gyr was the complex64 precision, fixed, see 7); gas loses ~3.6 %
   of its angular momentum per orbit at N=128 (numerical viscosity of Rusanov, should fall with resolution);
   Bondi accretion uses the constant `gas.cs`; the code's `W` (periodic Poisson) is ~17 % off the isolated value in a
   20 kpc box, so use an isolated `W` for virial checks (isolated soliton: 2K/|W| = 1.004).

7. Wave-function precision (fixed 2026-10-26): psi was kept in complex64 (created as complex64, the in-place `*=` keeps
   the dtype), so every kick and drift rounded to float32 and the norm of psi drifted proportionally to the number of
   steps (about -4e-5 % per step; -0.3 % per Gyr at h=1.25e-4, N=256; energy drift +0.28 % -> +1.2 % when h shrank 4x).
   Now `complex128` (`Wave_function_class.py`, `Wave_Packet_Class.py`): the norm is conserved to rounding and the results no
   longer depend on h. Results changed by ~1e-7 (relative). Only `baselines_koios_p100/` was regenerated; the Quadro, RTX
   and GTX sets are stale for ALL cases (psi dtype). psi snapshots/checkpoints are twice as big; old complex64 checkpoints
   stay complex64 on restart.
8. Particle tests (2026-10-26, `tests/`): `test_particle_kepler_orbit.py` (NBody.drift with one substep is the textbook DKD,
   mean order 2.06; the real step is set by the substep criterion 0.25*dx/v, not by h; production error <1e-3 r0 after 3
   orbits), `test_sink_force_vs_analytic.py` (sink force on the grid vs the Gaussian + periodic-box formula: mean error ~1e-4;
   the max error grows as (r/L)^4 above ~4 kpc = cubic lattice anisotropy, so the sink field is usable up to r <= L/4),
   `test_particle_breathing_soliton.py` (time-varying source: first order in h because the force is frozen at the start of
   the step; error ~1e-3 r0 for a source that changes 3x; accepted).
9. A circular orbit started with the analytic v = sqrt(GM/r) is NOT circular in a periodic box: at r = L/4 the force is ~6.5 %
   weaker (background term -(4 pi/3) G M r / L^3) plus 1-3 % axis anisotropy. Use r0 <~ L/10 or the speed from the grid force
   (`circular_velocity` in `test_particle_kepler_orbit.py`). `testing_baryons.py` is the minimal orbit script; the old orbit
   figure of the thesis (r0 = 20 kpc) shows exactly this effect.

10. Convergence with complex128 (main.py, phase 2, 1 Gyr, 2026-10-26; h = 5e-4 for N = 64/128, 1.25e-4 for N = 256; N = 32
   relaxes but the gas goes NaN in phase 2, too coarse): psi mass change ~1e-10 %, energy drift -0.036 / -0.024 / -0.015 % for
   N = 64 / 128 / 256 (before: +0.25 / +0.28 / +1.16 %, growing with the number of steps). Soliton converges: rho_max(t=0)
   1.500 / 1.508 / 1.511 e7, profile at r >= 1 kpc within 1 % between 128 and 256; below ~0.5 kpc slower (order ~1). Breathing
   of rho_max (10.3 / 12.2 / 13.4 % for N = 64 / 128 / 256) is physical: the soliton oscillates after the baryons are
   switched on. Its growth with N is mostly the better resolved peak (max 1.662 / 1.704 / 1.727e7, increments 0.042 / 0.023),
   not a different physics. Gas: angular-momentum loss -30.9 / -18.2 / -9.9 %
   (falls slowly, apparent order ~0.6 from three points), vphi at 1-3 kpc 5.28 / 6.34 / 6.55 (128 vs 256 within 3 %), outer
   gas (3-6 kpc) not converged. complex128 costs nothing measurable on the P100 (N = 256: 2 h 46 vs 2 h 42). The remaining
   non-converged part is the gas numerical viscosity (single forward-Euler MUSCL stage).

11. Sink-sink gravity (added 2026-10-27): sinks used to move only in `forces_environment` (no sink potential, to avoid self-force),
   so they did not attract each other. `SinkNBody._pair_accelerations` now adds a direct pair force, applied through the
   `_interpolate_force_trilinear` override (only for N_sinks > 1; a single sink is bitwise unchanged). Each sink is a Gaussian
   source (BH part softening_bh, reservoir part softening_cusp) that the others feel as a point mass (the kernel of the sink
   potential and of the energy log; symmetric and momentum-conserving unless sinks carry different reservoir fractions),
   periodic minimum image, plus the mean-density term -(4 pi/3) G m r / V of the periodic
   Poisson solver (far field matches the grid; valid up to r ~ L/4). Test `tests/test_sink_binary_kepler.py` (circular binary:
   separation 1.6e-4, angular velocity 2.3e-4, momentum and angular momentum conserved; it failed before the change).
   P100 reference set and optimization checks stay bitwise identical (no reference case has 2+ sinks). Demos with plots:
   `tests/sink_grav1.py` (infall from rest, merge at 3.888 Gyr vs 3.89 expected for 2 x 1e7 Msun, 10 kpc, box 30 kpc) and
   `tests/sink_grav2.py` (circular orbit, period ~23 Gyr). Merging: `merge_close_sinks` exists but is only called from the
   disabled sink-formation path; the demos call it after every step. It averages positions without periodic wrapping.
   The kernel was first a symmetric s^2 = 2 eps^2 one; changed the same day to the point-receiver kernel so that the force is
   the gradient of the potential energy Diagnostics logs. Energy conservation along a binary orbit is not measured yet.
12. Sink merging (2026-10-27): `sim.sink_merging = True` (and optionally `sim.sink_merge_radius` in kpc, default = capture radius)
   makes Evolution call `SinkNBody.merge_close_sinks` before every step (`Evolution._perform_sink_merging`); off by default.
   It is a resolution-scale rule, not merger physics: bound sinks closer than the merge radius become one sink at their
   centre of mass (mass, momentum conserved; no recoil, no gravitational-wave loss). Three gaps fixed: (a) the energy lost in
   a merge (kinetic + softened potential, same model as Diagnostics) is added to `E_diss_merge_total` and Diagnostics now
   includes it in `E_diss` (before, `E_tot_cons` jumped at every merge); (b) the centre of mass uses the minimum-image
   displacement and is wrapped into the box (a pair across the box edge used to merge in the middle of the box); (c) the bound
   check uses the softened escape speed 2 G M erf(r / (sqrt(2) eps)) / r instead of the point-mass one. Test
   `tests/test_sink_merging.py` (written first, failed on the old code; the residual jump of E_total + E_diss at a merge is
   ~5 % of the energy lost, the grid-deposited vs analytic sink energy; tolerance 10 %). It cleans up its output folders.
   The merge radius sets when the merge happens, so report the sensitivity to it (and to N) in any result. Merging is
   cheap: a few small GPU->CPU copies per call, well under 0.1 % of a step.
13. Dynamical friction test (2026-10-27): `tests/sink_friction.py` sends a sink (3e6 Msun, 30 kpc/Gyr) through a uniform psi
   (3e5 Msun/kpc^3, m = 2.5e-22 eV, quantum Mach number M_Q = 17.4, 19.2 kpc box, half a box of travel) and measures the drag
   coefficient C = F / F_rel, F_rel = 4 pi rho (G M / v)^2, from the sink's dv/dt. References: (1) exact linear theory for this
   very box (discrete mode sum: periodic, Gaussian sink of width `softening_bh`, switched on at t = 0, psi self-gravity included;
   u'' + Omega^2 u = -psi0 k^2 Phi_k exp(-i k.v t), Omega^2 = hbar^2 k^4 / 4 - 4 pi G rho0), (2) Lancaster et al. 2020
   (arXiv:1909.06381) Eq. 4.26 for a point source in an infinite medium with b = v t (formula checked against the paper text).
   Mean C for v t / lambda > 7, simulation / linear theory / literature: softening 1.5 cells 2.70 / 2.71 / 3.27, 1.0 cell 2.95 / 2.97,
   0.5 cell 3.17 / 3.17, and N = 256 with 0.25 cell (run by the user) 3.28 / 3.25 / 3.27; rms difference simulation vs theory
   0.5-1.1 %. The theory itself changes by < 0.3 % below 0.25 cell, so smaller softening is pointless. Rule of thumb:
   softening_bh <~ 0.3 lambda (lambda = hbar / (m v)) and lambda >= 3 cells. Not covered: the nonlinear regime M_Q <~ 1,
   a soliton background, conservation of the total momentum sink + psi (the sink feels the wake at a point, psi feels the
   Gaussian sink). The N = 256 run used H = 4e-4, above the printed stability limit (1.5e-4), with no visible harm.
14. Friction scans (2026-10-29): `tests/sink_friction_scan.py` (SCAN = "mq" or "softening", N = 128, plots in
   `resources/data/sink_friction_scan_<SCAN>/`). T1: mean drag coefficient for v t / lambda > 7 at M_Q = 100, 30, 10, 3, 1, 0.3
   (softening 0.29 lambda): simulation 3.148, 3.159, 3.185, 3.130, 2.788, 2.222; exact linear theory for the box 3.174, 3.174, 3.174,
   3.152, 3.132, 3.125; exact steady-state point-source formula (Lancaster Eq. 4.21, own implementation, tends to Eq. 4.26 for large M_Q)
   3.267, 3.271, 3.275, 3.216, 2.783, 1.923. Simulation = linear theory within 0.8 % for M_Q >= 3; at M_Q <= 1 the coefficient oscillates
   strongly (mean equals the steady state at M_Q = 1, 15 % above it at 0.3). T3: the momentum gained by psi differs from the momentum lost by
   the sink by -13.0 / -8.0 / -1.9 / +0.3 % for softening 0.86 / 0.57 / 0.29 / 0.14 lambda (the sink feels the wake at a point, psi feels the
   Gaussian sink). Fix implemented 2026-10-29 (uncommitted): `SinkNBody(smooth_force=True)` filters the force on the sink with the same Gaussian,
   momentum balance error drops to +0.3...+2 % for all softening and M_Q (test `tests/test_sink_force_smoothing.py`, 4.6e-1 -> 2e-15); default False;
   friction at finite softening is lower (Gaussian squared in the linear theory), same point-source limit. Details and caveats in `plans.md`.

15. Time-step check (2026-10-30, commit 1513a21): `Simulation_Class.check_time_step_restriction` only printed a warning above 0.5 x the stability limit
   (4 / (3 pi) dx^2 / hbar_tilde); `sim.strict_time_step = True` now raises ValueError instead (off by default, test `tests/test_time_step_strict.py`, registered
   in `run_all.py`). The criterion ignores the potential of a black hole (static_potential only); the BH phase H V_BH(0) / hbar_tilde < 0.1 is checked only in `correct_params.py`.
16. Black hole in a soliton and cross-code checks (2026-10-29..11-02; numbers and caveats in `plans.md` sections 8-11): T5 (`tests/t5_bh_soliton.py`, `tests/t5_analyze.py`,
   `slurm/koios_t5.sbatch`) reproduces the stopping time of arXiv:2403.09038 to ~2 % (45.9 vs 46.9 Myr, one run) but shows E_total drift 0.6-0.8 % and a growing psi + BH momentum (cause open);
   a quiet stationary BH needs r_B >= ~4-6 cells (rule of `correct_params.py`, not yet confirmed by a long run); the point receiver breaks down below ~0.5 cell; jaxion and our code
   agree on the drag to 0.2-2.6 % (`docs/crosscode_comparison/`, agent-written scripts are not reviewed); spin tests (`tests/spin/`, 5 scripts, not in `run_all.py`) pass on the P100.

### Accepted limitations (decided by the user, 2026-10-16)
- The internal (thermal) energy of **cold gas in an unresolved potential** is not reliable: the difference between
  the face-flux gravity work and the cell-centred work lands in `E−K`; a guard keeps it above the floor and
  tracks the injected energy in `gas.E_guard_injected` (~1.5 % of |E0| in the disk test). This is a property of
  the chosen solver; gas thermodynamics is not the priority.

### Session log — 2026-10-16 (what was done; details in `CONTINUATION.md` and `git log`)
- Moved to a new machine (Quadro P4000), created `baselines_quadro_p4000/`; later regenerated its gas cases.
- Found that gas gravity did not conserve energy (+3.6 |E0| in 0.06 Gyr). Fixes in `NBodyGas.py` / `Evolution_Class.py`:
  gravity work through cell faces, refreshed potential (kick–drift–kick), internal-energy guard; and in
  `Diagnostics_Class.py` gas internal energy was counted once per baryonic system → once. Drift now ~0.1 % (N=64).
- Verified against analytic cases: Sod shock tube, Jeans instability (growth and oscillation), free-fall collapse
  of a cold sphere, work/energy balance. Old code did not converge with resolution on the disk run; the new does.
- Fixed the heavy-collapse NaN (stale CFL step, see above). Cost of the new machinery: a few percent per step.
- Added tests: work balance, positive internal energy, U_iso counted once, Sod, Jeans, free fall, vacuum cell,
  adaptive substeps. `tests/` is no longer git-ignored.

### What I (Claude) learned this session — apply next time
- **Measure before explaining.** Several first hypotheses were wrong (e.g. "floors cause the NaN", "old code
  converges in Jeans"). The decisive tools were: instrument per operation, compare old vs new by monkey-patching
  (`Evolution._set_gas_potential = lambda self, a, b, c: None` reproduces the old gas behaviour bitwise),
  and compare with an exact solution.
- **Check the test, not only the code.** Jeans first failed because my initial pressure was isothermal instead of
  adiabatic; Sod first failed because periodic waves met at t=0.15. When a validation test disagrees, suspect
  the setup first, then the code.
- A constant offset in a measured growth rate can be a time-label offset; use a local log-slope instead.
- Redirected stdout swallows your own prints; filtering output with grep can hide the one line you need.
- Report failures of your own change plainly (the new code gave NaN where the old one gave a clean error).
- Energy diagnostics can be wrong independently of the dynamics (the double counted U_iso): separate
  "dynamical energy" from the logged `E_total` before blaming the solver.
- Periodic box: a rotating disk IC is not periodic; the "cross/arms" in density slices come from box faces.
- Collapse planning: t_ff = sqrt(3π/(32 G ρ0)); use h ≈ t_ff/100; `max_substeps` may need raising.
- Do not leave simulation output behind; `resources/data` fills the disk (8 GB removed today).

### Session log — 2026-10-26 (Koios, details in `CONTINUATION.md`)
- Running on the Koios cluster: see `KOIOS.md` (ssh-agent, venv, sbatch). P100 reference set `baselines_koios_p100/`
  is bitwise reproducible; `compare_baseline_sets.py` compares two sets.
- Fixed: window-disk CFL case (see issue 1), sink potential offset and accretion energy (issue 5). New tests in
  `tests/` for all three; `run_all.py` passes.
- `main.py` is now a simplified dwarf galaxy (soliton 5.3e7 Msun + stars + gas + black hole, masses as fractions of
  the soliton) in two phases: relaxation with stars switched on slowly, then `from_checkpoint` + `start_new_segment`
  with gas and sink. Plain relaxation does not damp the soliton breathing (+-13 %, period ~0.75 Gyr); slow switching-on does.
- `plot_energies.py` / `plot_density.py` / `Plot_max_vals.py` / `plot_from_snapshots.py` save images into the simulation directory.
- Nothing is committed.

### Session log — 2026-10-26 (evening)
- Convergence runs N=64/128/256 (phase 2, 1 Gyr, complex64): soliton profile and inner gas rotation converge (128 vs 256
  within 1-2 %), gas angular momentum loss -31/-18/-10 %. Energy drift and psi mass loss grew with the NUMBER OF STEPS, not
  with N: cause = complex64 psi (issue 7). Fixed with complex128; `tests/run_all.py` passes; P100 reference set regenerated.
- New tests for particles and the sink field (issue 8); minimal orbit script `testing_baryons.py` (analytic Kepler speed).
- Next: convergence N=32/64/128/256 with complex128; sink oscillation test in the soliton; open points 2 and 3 above.

### Research plan (paper idea, literature numbers, test list, results, conclusions): `plans.md` (git-ignored on purpose, the repository is public: it exists only on Koios, keep a backup; sections 11.x = conclusions and open work)

### Handoff 2026-10-27 — next tasks (read this first in a new session)
Goal of the next phase: **black-hole collisions inside a soliton** (ULDM soliton 5.3e7 Msun, sinks as black holes). Gas is NOT
needed for this (drop it; its numerical viscosity and outer-ring problems are then irrelevant).

**RESOLVED later on 2026-10-27 (issue 11 above): sink-sink gravity is implemented and tested. Text below is the original diagnosis.**
**Blocker found (2026-10-27): sinks do not attract each other.** Two sinks of 5e10 Msun, 4 kpc apart, no psi, 0.1 Gyr: the
separation stays exactly 4.0000 kpc (Newton would collapse them in ~0.01 Gyr). Sinks move only in `forces_environment`
(force from the density without any sink potential, `Evolution._evolve_order_2`), so there is no sink-sink gravity. Reproducer:
`~/conv/exp_twosinks.py` (run on a GPU node with `PYTHONPATH=$PWD`). Only this configuration (no psi) was measured; the cause
in the code was not investigated further. Changing it is a physics change: explain first, say which reference sets change.

Ordered plan (agreed with the user):
1. DONE (issue 11): add sink-sink gravity (softened, consistent with the analytic sink potential), plus a regression test with a two-sink
   Kepler binary (period, energy, centre of mass). Style of the existing tests: `tests/test_particle_kepler_orbit.py`.
2. Dynamical-friction test: a sink moving through psi (uniform background or soliton) against the literature formula (Hui et
   al. 2017, Lancaster et al. 2020 — recalled from memory, verify before using). This also becomes the validation set for GAMER.
3. Resolution check: BH of 2e5 Msun (0.4 % of the soliton) dominates only inside ~150 pc (rough estimate, constant core density)
   = 2 cells at N = 256 (dx = 78 pc). A BH of ~1e7 Msun dominates inside ~0.5 kpc (resolved). Prototype with the heavy BH.
   Uniform N = 512 would cost ~60 h per Gyr on a P100 (extrapolated from N = 128 -> 256: 7.4 min -> 166 min), so a scan needs AMR.
4. GAMER (https://github.com/gamer-project/gamer): GPU + AMR + ELBDM (wave and hybrid schemes, refinement flag for
   interference). Its wiki does not document ELBDM + particles or BH sinks (sink particles exist only for gas in a 2025 paper):
   ask the developers on GitHub before deciding. Koios has no `nvcc` (try pip CUDA toolkit wheels or Phoebe). Decide after 1-3.
5. Maintenance so the code is "functional/publishable": push (see below), `requirements.txt` is UTF-16, `.idea` is tracked,
   duplicate `system_fucntions.py`, README (units, minimal example), licence and citation, self-interaction potential (point 3:
   fix or switch off and document), regenerate the Quadro/RTX/GTX reference sets on Windows (stale for ALL cases since complex128).

Physics notes for this phase: an isolated soliton is stationary; the ~10 % breathing seen in main.py is a coherent mode excited by
the baryons (physical, user's view). The random "soliton walk" appears only if the soliton sits in a halo with interfering
waves; main.py has no such halo. A circular orbit with v = sqrt(GM/r) is not circular in the periodic box (use r0 <~ L/10 or the
grid force); the sink field is usable up to r <= L/4.

State of the repo and data (Koios, `~/projects/research_task`, branch `working_with_chat`):
- Commit `0d11459` (tests, minimal orbit script, docs) and `a236daa` (complex128) exist ONLY on Koios: `git push` over https needs
  a GitHub token (the user does it). Uncommitted: the convergence notes in CLAUDE.md/CONTINUATION.md and this handoff,
  `tests/test_sink_oscillation_in_soliton.py` (passes, registered in `tests/run_all.py`), untracked `slurm/koios_*_c128.sbatch`,
  `koios_convergence_h.sbatch` (one-off scripts; do not commit).
- `resources/data/` holds ~54 GB (home has no backup). Proposed deletions (not yet approved), exact names in
  `~/conv/delete_tier_A.txt` (34 small test runs, 0.6 GB), `_B.txt` (8 older exploratory runs, 8.6 GB), `_C.txt` (4 superseded
  complex64 N = 256 / N = 128 h = 1.25e-4 runs, 18 GB). Delete only by exact names the user approves.
- Helper scripts (outside the repo, in `~/conv/`): `conv_metrics.py` (convergence table), `plot_particle_tests.py` (plots of tests
  A-C), `plot_sink_osc.py`, `exp_twosinks.py`. Plots from this session are in `plots/particle_tests/` on the Windows machine.
- Connection: PowerShell `ssh koios` (the Windows ssh-agent already holds the key). Long jobs: submit with `sbatch` (a long
  `srun` over ssh drops the connection); only one P100 is usually free, so do not leave an interactive `srun --pty bash` open.

