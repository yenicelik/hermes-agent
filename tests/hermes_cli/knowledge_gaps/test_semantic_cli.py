"""The public comparison CLI feeds JSON to existing review systems."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hermes_cli.knowledge_gaps.core import text_hash

ROOT = Path(__file__).resolve().parents[3]


class SemanticCliTests(unittest.TestCase):
    def fixture(self):
        scope = dict(account='example-account', tenant=None, policy_revision='policy-1',
                     coverage='bounded', public_external_allowed=False)
        sources = []
        for ident, kind in [('internal-note', 'internal'), ('public-paper', 'external')]:
            text = 'The review needs a measured recovery example.'
            sources.append(dict(id=ident, kind=kind, title=ident,
                                uri='https://example.test/'+ident, text=text,
                                sha256=text_hash(text), revision='revision-1',
                                account='example-account', tenant=None, policy_revision='policy-1',
                                authorized=True, current=True, withdrawn=False, access='accessible',
                                capture='captured', index='indexed', published_at=None,
                                observed_at='2026-01-01T00:00:00Z'))
        envelope = dict(schema_version='hermes.gaps.input.v1', as_of='2026-01-01T00:00:00Z',
                        scope=scope, sources=sources)
        refs = [dict(source_id=s['id'], start=0, end=len(s['text']), quote=s['text'],
                     sha256=s['sha256'], revision=s['revision']) for s in sources]
        proposals = [dict(id='recovery-example', kind='missing_connection',
                          title='Connect the recovery evidence', observation='A proposed connection.',
                          why_it_matters='It may answer an open question.', next_step='Review the sources.',
                          uncertainty='Not proven.', impact=2, evidence=refs, basis='agent_review')]
        return envelope, proposals

    def invoke(self, folder, extra=()):
        return subprocess.run([sys.executable, '-m', 'hermes_cli.knowledge_gaps.semantic_cli',
                               '--input', str(folder/'input.json'), '--proposals',
                               str(folder/'proposals.json'), *extra], cwd=ROOT,
                              capture_output=True, text=True, timeout=10)

    def test_stdout_default_creates_no_output_files(self):
        envelope, proposals = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'input.json').write_text(json.dumps(envelope), encoding="utf-8")
            (folder/'proposals.json').write_text(json.dumps(proposals), encoding="utf-8")
            before = sorted(p.name for p in folder.iterdir())
            result = self.invoke(folder)
            self.assertEqual(result.returncode, 0, result.stderr)
            parsed = json.loads(result.stdout)
            self.assertEqual(parsed['counts']['candidates'], 1)
            self.assertEqual(parsed['status'], 'proposal_only')
            self.assertEqual(sorted(p.name for p in folder.iterdir()), before)

    def test_explicit_private_snapshot_and_hallucination_refusal(self):
        envelope, proposals = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'input.json').write_text(json.dumps(envelope), encoding="utf-8")
            (folder/'proposals.json').write_text(json.dumps(proposals), encoding="utf-8")
            result = self.invoke(folder, ['--output', str(folder/'review.json')])
            self.assertEqual(result.returncode, 0, result.stderr)
            original = (folder/'review.json').read_bytes()
            self.assertEqual((folder/'review.json').stat().st_mode & 0o777, 0o600)
            proposals[0]['evidence'][0]['quote'] = 'invented private text'
            (folder/'proposals.json').write_text(json.dumps(proposals), encoding="utf-8")
            failed = self.invoke(folder, ['--output', str(folder/'review.json')])
            self.assertEqual(failed.returncode, 2)
            self.assertNotIn('invented', failed.stderr + failed.stdout)
            self.assertEqual((folder/'review.json').read_bytes(), original)

    def test_removed_html_option_is_rejected(self):
        envelope, proposals = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'input.json').write_text(json.dumps(envelope), encoding="utf-8")
            (folder/'proposals.json').write_text(json.dumps(proposals), encoding="utf-8")
            failed = self.invoke(folder, ['--html', str(folder/'review.html')])
            self.assertEqual(failed.returncode, 2)
            self.assertFalse((folder/'review.html').exists())


if __name__ == '__main__':
    unittest.main()
