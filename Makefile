

.PHONY: test
test: 
	cargo watch -c -x 'test -- --nocapture'

.PHONY: run
run: 
	cargo watch -c -x 'run'

.PHONY: lint
lint:
	cargo watch -c -x 'clippy'

