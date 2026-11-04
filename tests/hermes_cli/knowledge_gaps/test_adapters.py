import copy
import unittest
from hermes_cli.knowledge_gaps.adapters import Selection, collect
from hermes_cli.knowledge_gaps.core import InputError, analyze, text_hash

ASOF = "2026-11-03T12:00:00Z"
SCOPE = dict(account="owner", tenant="research", policy_revision="policy-1",
             public_external_allowed=False, coverage="bounded")


def row(ident, kind="internal"):
    text = "Which retrieval cases lack verified evidence?"
    return dict(id=ident, revision="r1", sha256=text_hash(text), text=text,
                uri="https://example.test/source", title="Evidence review", kind=kind,
                account="owner", tenant="research", policy_revision="policy-1",
                authorized=True, current=True, withdrawn=False, access="accessible",
                capture="captured", index="indexed", observed_at=ASOF, published_at=ASOF)


class AdapterTests(unittest.TestCase):
    def test_invalid_private_domain_never_invokes_reader(self):
        calls = []
        def reader(key):
            calls.append(key)
            return row("internal-1")
        for account, tenant in (("other-account", "research"), ("owner", "other-tenant"), ("public", None)):
            with self.assertRaises(InputError):
                collect({"native": reader}, [Selection("native", "k", "internal-1", account, tenant, "internal")],
                        scope=SCOPE, as_of=ASOF)
        self.assertEqual(calls, [])

    def test_all_selections_validated_before_first_read(self):
        calls = []
        def reader(key):
            calls.append(key)
            return row("first")
        selections = [Selection("native", "k1", "first", "owner", "research", "internal"),
                      Selection("native", "k2", "second", "other", "research", "internal")]
        with self.assertRaises(InputError):
            collect({"native": reader}, selections, scope=SCOPE, as_of=ASOF)
        self.assertEqual(calls, [])

    def test_caller_mutations_cannot_redirect_recheck_or_domain(self):
        scope = copy.deepcopy(SCOPE)
        selections = [Selection("native", "k", "internal-1", "owner", "research", "internal")]
        readers = {}
        calls = []
        def reader(key):
            calls.append(key)
            scope["account"] = "changed-by-reader"
            selections.clear()
            readers["native"] = lambda _: self.fail("recheck redirected")
            return row("internal-1")
        readers["native"] = reader
        data, receipt = collect(readers, selections, scope=scope, as_of=ASOF)
        self.assertEqual(data["scope"]["account"], "owner")
        self.assertEqual(len(calls), 2)
        self.assertEqual(receipt["selection_count"], 1)

    def test_malformed_second_read_is_sanitized(self):
        calls = []
        def reader(key):
            calls.append(key)
            return row("internal-1") if len(calls) == 1 else {"private": float("nan")}
        with self.assertRaisesRegex(InputError, "recheck failed"):
            collect({"native": reader}, [Selection("native", "k", "internal-1", "owner", "research", "internal")],
                    scope=SCOPE, as_of=ASOF)

    def test_source_neutral_readers(self):
        # Different transports need no engine branches or author-name rules.
        for transport in ("notion_api", "git_markdown", "rss_article", "meeting_transcript", "mail_capture"):
            source = row("canonical-document")
            spec = Selection(transport, "opaque-native-key", source["id"], "owner", "research", "internal")
            data, receipt = collect({transport: lambda key: copy.deepcopy(source)}, [spec], scope=SCOPE, as_of=ASOF)
            result = analyze(data)
            self.assertEqual(result["counts"]["searchable_sources"], 1)
            self.assertTrue(receipt["all_rechecks_equal"])
            self.assertNotIn(source["text"], str(receipt))

    def test_withdrawal_during_read_refuses_snapshot(self):
        source = row("internal-1")
        calls = []
        def reader(key):
            calls.append(key)
            r = copy.deepcopy(source)
            if len(calls) > 1:
                r.update(withdrawn=True, text="", sha256=text_hash(""))
            return r
        with self.assertRaisesRegex(InputError, "changed"):
            collect({"native": reader}, [Selection("native", "k", "internal-1", "owner", "research", "internal")], scope=SCOPE, as_of=ASOF)

    def test_identity_redirect_refused(self):
        for key, value in (("account", "other"), ("id", "other"), ("kind", "external"), ("policy_revision", "p2")):
            r = row("internal-1"); r[key] = value
            with self.assertRaises(InputError):
                collect({"native": lambda _: r}, [Selection("native", "k", "internal-1", "owner", "research", "internal")], scope=SCOPE, as_of=ASOF)

    def test_provider_error_body_not_exposed(self):
        def reader(key):
            raise RuntimeError("PRIVATE PROVIDER BODY")
        with self.assertRaises(InputError) as caught:
            collect({"native": reader}, [Selection("native", "k", "internal-1", "owner", "research", "internal")], scope=SCOPE, as_of=ASOF)
        self.assertNotIn("PRIVATE", str(caught.exception))

    def test_same_input_byte_stable(self):
        r = row("internal-1")
        args = ({"native": lambda _: copy.deepcopy(r)}, [Selection("native", "k", "internal-1", "owner", "research", "internal")])
        self.assertEqual(collect(*args, scope=SCOPE, as_of=ASOF), collect(*args, scope=SCOPE, as_of=ASOF))


if __name__ == "__main__":
    unittest.main()
