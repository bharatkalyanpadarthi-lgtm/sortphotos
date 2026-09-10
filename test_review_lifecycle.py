"""Review completion is terminal, including delayed browser batch requests."""

import json
import os
import subprocess
import tempfile
import threading
import unittest
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import review_unknown_identities as review


class ReviewLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            'lifecycle_lock': threading.Lock(), 'lock': threading.Lock(),
            'action_queue': SimpleNamespace(is_idle=lambda: True),
            'batch_loading': False, 'finishing': False, 'review_finished': False,
            'quiet': True, 'items_by_key': {},
        }
        self.job = review.BatchLoadJob(self.state)
        self.state['batch_job'] = self.job

    def test_completed_review_batch_is_idempotent_without_worker(self):
        self.state['review_finished'] = True
        with patch.object(review.threading, 'Thread', side_effect=AssertionError('new worker')):
            for _ in range(5):
                payload, error = self.job.start()
                self.assertIsNone(error)
                self.assertEqual(payload['reason'], 'review_finished')
                self.assertEqual(payload['status'], 'completed')
                self.assertTrue(payload['complete'])
        self.assertFalse(self.state['batch_loading'])

    def test_finishing_and_queued_actions_have_distinct_conflicts(self):
        self.state['finishing'] = True
        payload, error = self.job.start()
        self.assertTrue(error)
        self.assertEqual(payload['reason'], 'finishing')
        self.state['finishing'] = False
        self.state['action_queue'] = SimpleNamespace(is_idle=lambda: False)
        payload, error = self.job.start()
        self.assertTrue(error)
        self.assertEqual(payload['reason'], 'actions_pending')
        self.assertIsNone(self.job.worker)

    def test_running_batch_is_reused_without_duplicate_worker(self):
        self.state['batch_loading'] = True
        self.job._set(status='running', processed=21)
        payload, error = self.job.start()
        self.assertIsNone(error)
        self.assertEqual(payload['processed'], 21)
        self.assertIsNone(self.job.worker)

    def test_finish_during_request_body_read_cannot_enqueue_or_skip(self):
        for route in ('/decide', '/skip'):
            with self.subTest(route=route):
                self.state['finishing'] = False
                self.state['items_by_key'] = {'item': object()}
                self.state['action_queue'].submit_many = Mock(side_effect=AssertionError('late submission'))
                handler = review.make_handler(self.state).__new__(review.make_handler(self.state))
                handler.path = route
                body = b'action=keep_unknown&item_keys=item'
                handler.headers = {'Content-Length': str(len(body))}

                def finish_while_reading(_length):
                    self.state['finishing'] = True
                    return body

                handler.rfile = SimpleNamespace(read=finish_while_reading)
                handler.send_json = Mock()
                handler.do_POST()
                self.assertEqual(handler.send_json.call_args.args[1], 409)
                self.state['action_queue'].submit_many.assert_not_called()
                self.assertNotIn('temporarily_skipped', self.state)

    def test_http_finished_review_rejects_mutations_and_returns_terminal_batch(self):
        self.state['review_finished'] = True
        self.state['action_queue'].submit_many = Mock(side_effect=AssertionError('mutation'))
        server = ThreadingHTTPServer(('127.0.0.1', 0), review.make_handler(self.state))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f'http://127.0.0.1:{server.server_port}'
            with urlopen(Request(base + '/next-batch', method='POST')) as response:
                self.assertEqual(response.status, 202)
                self.assertEqual(json.load(response)['reason'], 'review_finished')
            for route in ('/skip', '/decide'):
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base + route, method='POST'))
                self.assertEqual(error.exception.code, 409)
            self.state['action_queue'].submit_many.assert_not_called()
        finally:
            server.shutdown()
            thread.join(3)
            server.server_close()

    @unittest.skipUnless(os.environ.get('FACE_REVIEW_BROWSER_TEST') == '1',
                         'Set FACE_REVIEW_BROWSER_TEST=1 with Node/Playwright to run browser regressions')
    def test_browser_lifecycle(self):
        # Reuse the synthetic, non-personal item fixture used by explanation tests.
        from test_review_explanations import ReviewExplanationTests
        fixture = ReviewExplanationTests()
        fixture.setUp()
        try:
            item = replace(fixture.item(), key='fixture')
            cluster = review.UnknownCluster('fixture', (item,), item.candidates, 0.)
            with tempfile.TemporaryDirectory(prefix='review-browser-') as directory:
                root = Path(directory)
                for name, clusters in [('empty', []), ('pending', [cluster])]:
                    (root / f'{name}.html').write_text(review.render_html(
                        clusters, {}, ['Alice', 'Bob'], {}, interactive=True))
                result = subprocess.run([
                    os.environ.get('NODE_BINARY', 'node'),
                    str(Path(__file__).with_name('review_lifecycle.browser.cjs')), str(root),
                ], capture_output=True, text=True, timeout=150)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                print(result.stdout)
        finally:
            fixture.doCleanups()


if __name__ == '__main__':
    unittest.main()
