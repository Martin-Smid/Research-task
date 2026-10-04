# Pokračování práce — working_with_chat

**Nejdřív zjistit, co je problém u CFL v NBodyGas.** Zatím není určena příčina ani ověřená oprava. Reprodukovat konkrétní běh, zkontrolovat odhad stabilního kroku, subcycling a limit max_substeps; neměnit fyziku naslepo.

## Cíl a pravidla

Zjednodušit simulátor, omezit zbytečné výpočty a závislosti a připravit dokumentaci pro publikování. Zachovat strukturu autorova kódu a malé, čitelné změny. Při každé úpravě numerického výpočtu porovnat mřížky, částice a energie s původními referencemi; změna výsledků znamená zastavit a vyšetřit příčinu. Referenční data nepřepisovat jako způsob opravy testu.

`working_with_chat` obsahuje testy, reference a historii v resources/diagnostics; main má stejný produkční kód a příklady bez tohoto vývojového adresáře a chats_playground.py. Tento soubor patří pouze do pracovní větve. Na novém počítači začít touto větví, přečíst resources/diagnostics/README.md a docs/rotation_curves.md, připravit CUDA/CuPy prostředí a ověřit testy před změnami.

## Co už je hotové

- Gravitace: stejný potenciál pro wave kick a baryon kick se znovu použije pouze v rámci stejného integračního stavu. Konstantní jmenovatel Poissonova výpočtu se připraví jednou v Propagator; pořadí kroků a aktualizace hustoty se zachovaly. Autor naměřil přibližně 40–45 -> 50–55 FPS.
- Diagnostics_Class.py: Evolution předává aktuální stav a poskytovatel sink potenciálu, Diagnostics počítá energie/profily a organizuje rotace; výsledky jdou přes Evolution do Scribe. Historická vnořená akumulace plynové energie byla záměrně zachována; případnou opravu řešit samostatně jako změnu fyzikálních výsledků.
- Rotation_Curve_Class.py: společný GPU výpočet pro plyn a částice, periodický hmotnostní střed, odečtení pohybu těžiště a osa z celkového momentu hybnosti. Výchozí cutoff je válcový poloměr obsahující 50 % hmotnosti, ne 50 % maximální hustoty; neurčená osa vrací stav místo smyšlené křivky. NBody/NBodyGas volají společnou třídu, Diagnostics vrací malé profily a metadata do Evolution, Scribe zapisuje rotational_velocity.dat a rotation_frames.csv.
- plot_rot_curves.py: bez argumentů poslední běh a oba typy komponent; -g/-N filtr, více adresářů překryje více běhů, --no-show pouze uloží. examples/plot_rotation_curves.py sdílí implementaci.
- Úklid: odstraněno 188 simulací před 2026-10-04 (1,86 GiB); dnešní běhy a reference zachovány. Jednorázové migrační skripty jsou v diagnostickém archivu a nelze je přímo znovu spustit.

## Ověření a omezení

Původní čtyři případy měly bitově shodné mřížky a energie; rozšířené porovnání optimalizace zahrnovalo 12 případů. Oddělení diagnostik a rotace ověřeno v sedmi případech s 49 energetickými vzorky. Restart bitově shodný; drobné segmentové rozdíly existovaly už ve starém kódu. Osm analytických testů rotací prošlo i po úklidu. Povolené změny uložených výstupů po nové rotaci jsou pouze rotational_velocity.dat a rotation_frames.csv.

NBodyGas podporuje řád evoluce 2; řády 4/6 mají záporné podkroky a jsou odmítnuty. Plynový dt musí být konečný a nezáporný. Nezaměňovat tato omezení s dosud nevyšetřeným problémem CFL. Nakloněný disk N=128 se správnou osou není důkazem fyzikální rovnováhy.

```powershell
python chats_playground.py --help
python chats_playground.py compatibility-check
python resources/diagnostics/rotation_playground.py
python resources/diagnostics/check_optimization.py --help
```

Pro porovnání rotací použít podporovaný přepínač --allow-rotation-changes podle nápovědy playgroundu. Lokální .venv není přenosná a data simulací jsou ignorovaná; uložené referenční NPZ a manifesty v pracovní větvi jsou přenosné. Na Windows může být nutné použít .venv/Scripts/python.exe místo systémového aliasu python.

## Další práce po CFL

1. **Reduce structural clutter.** Consolidate the two system_fucntions.py modules after checking their callers, replace wildcard imports with explicit imports, and move plotting helpers out of engine dependencies. Extract the repeated baryon advancement code while preserving the integration sequences.
2. **Make installation reproducible.** Add package metadata, distinguish runtime dependencies from plotting/development dependencies, and verify a clean installation. One concrete gap: the engine imports tqdm, but requirements.txt omits it. Tracked editor files and historical outputs also need a deliberate cleanup.
3. **Document the verified behavior.** Replace the README's development diary with installation, a minimal runnable example, units, boundary conditions, supported components, and restart instructions. Add numerical-method documentation, checkpoint-format documentation, and reproducible validation cases with stated tolerances. Before release, settle licensing and citation information with the author.

Postupovat po malých změnách, nejprve měřit a porovnávat výsledky. Vývojové experimenty držet v working_with_chat; do main přenášet ověřený kód, příklady a uživatelskou dokumentaci.
