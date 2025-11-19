.PHONY: data verify clean

OUTDIR ?= .
LEDGER ?= v5_ledger.json

data:
	./generate_all.py --outdir $(OUTDIR) --seed_ledger $(LEDGER)

verify:
	@echo "Verifying checksums (v5 preferred; fallback to .sha256)…"
	@if [ -f "$(OUTDIR)/checksums_v5.txt" ]; then sha256sum -c "$(OUTDIR)/checksums_v5.txt"; else sha256sum -c "$(OUTDIR)/checksums.sha256.txt"; fi

clean:
	@echo "Cleaning generated tables and manifests from $(OUTDIR)…"
	@rm -f $(OUTDIR)/datatable*.dat
	@rm -f $(OUTDIR)/checksums_v5.txt $(OUTDIR)/checksums.sha256.txt
	@rm -f $(OUTDIR)/*.json
