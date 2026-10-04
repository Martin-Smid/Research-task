# Přehled diagnostik a testovacích souborů

`resources/data` obsahuje výstupy simulací. Zde jsou pomocné testy, zmrazené reference a historie úprav; žádný z těchto nástrojů se nespouští při běžné simulaci.

Zdrojové soubory, dokumentace, referenční NPZ a souhrnné JSON nejsou ignorovány Gitem. Logy, obrázky, cache, velké checkpointy a generované restart/segment běhy jsou lokální a ignorované; přesun sám o sobě nic necommitoval.

## Používané nástroje

| Soubor | Co dělá |
| --- | --- |
| `../../chats_playground.py` | Hlavní testovací vstup: porovnává stav mřížek, částic a energie s uloženými referencemi a ověřuje restart i navazující segmenty. |
| `check_optimization.py` | Porovnává 12 případů před/po optimalizaci gravitace a měří čas a počet výpočtů Poissonovy rovnice. |
| `verify_old_segments.py` | Spouští historické třídy a zjišťuje, zda malé rozdíly navazujících segmentů existovaly už před optimalizací. |
| `rotation_playground.py` | Osm analytických kontrol rotačních křivek: naklonění, periodické hranice, hmotnostní váhy, poloměr, plyn a neurčená osa. |
| `../Classes/Diagnostics_Class.py` | Počítá energie, radiální profily a organizuje rotační diagnostiku z aktuálního stavu evoluce. Výstupy nadále zapisuje Scribe. |
| `../Classes/Rotation_Curve_Class.py` | Společný GPU výpočet středu, pohybu těžiště, rotační osy a hmotnostně vážených křivek pro plyn i částice. |
| `../../plot_rot_curves.py` | Vykreslí poslední nebo zadané běhy, oba typy komponent nebo výběr `-g`/`-N`, včetně disperze rychlosti. |
| `../../examples/plot_rotation_curves.py` | Krátký kompatibilní vstup do stejného plotteru; neobsahuje druhou kopii výpočtu. |
| `../../docs/rotation_curves.md` | Definice osy, středu, poloměru obsahujícího polovinu hmotnosti, jednotek a použití API i plotteru. |

Z kořene projektu lze spustit například:

```powershell
python resources/diagnostics/rotation_playground.py
python chats_playground.py compatibility-check
python plot_rot_curves.py
python plot_rot_curves.py -g
python plot_rot_curves.py -N simulation_20261004_100928_726693
python plot_rot_curves.py simulation_20261004_102145_365587 simulation_20261004_100928_726693
```

## `archive/`: historické kopie a jednorázové úpravy

Tyto soubory vysvětlují historii změn; nejsou součástí běžného workflow. Přímé spuštění je zablokované, protože migrační skripty by znovu přepisovaly zdrojové soubory; historické třídy lze stále importovat pro porovnání.

| Soubor | Co dělá |
| --- | --- |
| `edit_rotation_curves.py` | Jednorázově vytvořil společný výpočet rotačních křivek a nahradil původní metody komponent jeho voláním. |
| `extract_diagnostics.py` | Jednorázově přesunul diagnostické výpočty z Evolution do Diagnostics a ponechal kompatibilní vstupní metody. |
| `tidy_diagnostics.py` | Po extrakci odstranil přebytečné komentáře a prázdné řádky; neměnil fyzikální výpočty. |
| `relocate_diagnostics.py` | Jednorázově přesunul testovací soubory z data sem a upravil jejich cesty a ignorování generovaných dat. |
| `summarize_diagnostics.py` | Vytvořil souhrn kontrol po oddělení diagnostik, před změnou rotačních křivek. |
| `summarize_rotation.py` | Vytvořil souhrn kontrol po zavedení společného výpočtu rotačních křivek. |
| `evolution_before_diagnostics.py` | Kopie Evolution po optimalizaci gravitace, ale před oddělením diagnostik; dokumentuje původní umístění výpočtů. |
| `playground_before_diagnostics.py` | Kopie staršího testovacího nástroje před rozšířením o diagnostické benchmarky. |
| `old_evolution.py` | Původní Evolution před optimalizací gravitace, používaná při historickém porovnání segmentů. |
| `old_propagator.py` | Původní Propagator před optimalizací Poissonova výpočtu, používaný ve stejném porovnání. |

## `reports/`: malé souhrny a kontrolní obrázky

| Soubor | Co obsahuje |
| --- | --- |
| `cleanup_report.json` | Datum úklidu, seznam 188 smazaných starších simulací a jejich celkovou velikost. |
| `diagnostics_original_hashes.json` | SHA256 původních referencí; umožňuje ověřit, že nebyly dodatečně změněny. |
| `diagnostics_report.json` | Historický souhrn oddělení diagnostik: sedm případů, 49 energetických vzorků a porovnání uložených výstupů. |
| `rotation_report.json` | Souhrn změny rotačních křivek, analytických testů a shody mřížek i energií; pouze rotační výstupy se směly změnit. |
| `rotation_curve_preview.png` | Původní kontrolní obrázek rotační křivky částicového prstence. |
| `plot_gas.png` | Kontrola nového plotteru při výběru pouze plynu. |
| `plot_nbody.png` | Kontrola nového plotteru při výběru pouze částic. |
| `plot_comparison.png` | Kontrola překrytí křivek z více simulací v jednom obrázku. |

## `baselines/` a `runtime/`

| Adresář / soubory | Účel |
| --- | --- |
| `baselines/chat_regression_baselines/*.npz` a `manifest.json` | Zmrazené původní referenční stavy čtyř základních případů a jejich parametry. Nepřepisovat při běžném porovnávání. |
| `baselines/chat_regression_baselines/diagnostics/*.npz` a `manifest.json` | Sedm referenčních případů s diagnostikami, energetickou historií a kontrolními součty uložených výstupů. |
| `baselines/optimization_checks/` | Dvanáct referencí optimalizace pro různé řády evoluce a komponenty, plus historický report segmentů. |
| `baselines/checkpoint_diagnostics/` | Zachované lokální experimenty s checkpointy; velká generovaná data se neposílají na GitHub. |
| `restart_runs/`, `segment_runs/`, `latest_*.json` pod baselines | Nově generované replay výstupy a poslední podrobné výsledky testů; nejsou zmrazené reference. |
| `runtime/` | Lokální cache GPU kompilace a dočasné soubory testů; nejsou součástí simulátoru ani referencí. |

## `logs/`: výpisy jednotlivých ověření

Logy slouží pro dohledání detailu; samotné shrnutí je v reports. Každý vznikl spuštěním příslušného testu, nikoli při běžné simulaci.

| Soubor | Význam |
| --- | --- |
| `before_optimization.log` | Původní běhy před optimalizací. |
| `after_optimization.log` | Porovnání základních běhů po optimalizaci. |
| `optimization_before.log` | Vytvoření rozšířených referencí optimalizace. |
| `optimization_after.log` | Porovnání rozšířených případů po optimalizaci. |
| `optimization_compatibility.log` | Kontrola povolených řádů evoluce a plynových kroků. |
| `optimization_old_segment.log` | Segmentové porovnání se starými třídami. |
| `optimization_restart.log` | Kontrola restartu po optimalizaci. |
| `optimization_segment.log` | Kontrola navazujících segmentů po optimalizaci. |
| `diagnostics_before.log` | Referenční diagnostické běhy před extrakcí. |
| `diagnostics_after.log` | Porovnání diagnostik po extrakci. |
| `diagnostics_original_compare.log` | Ověření původních základních referencí po extrakci. |
| `diagnostics_compatibility.log` | Kontrola kompatibility solveru po extrakci. |
| `diagnostics_restart.log` | Kontrola restartu po extrakci. |
| `diagnostics_segment.log` | Kontrola segmentů po extrakci. |
| `rotation_analytic.log` | Osm analytických kontrol nového výpočtu rotačních křivek. |
| `rotation_diagnostics.log` | Porovnání mřížek a energií po změně rotačních diagnostik. |
| `rotation_original_compare.log` | Ověření původních základních referencí po změně rotací. |
| `rotation_restart.log` | Kontrola restartu po změně rotací. |
| `rotation_segment.log` | Kontrola segmentů po změně rotací. |
| `rotation_after_cleanup.log` | Zopakování osmi analytických kontrol po přesunu souborů. |
| `compatibility_after_cleanup.log` | Zopakování kompatibility solveru po přesunu souborů. |
