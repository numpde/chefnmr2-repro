"""Adversarial Make inputs and authority rejection behavior."""

from operator_fixtures import OperatorFixture, ROOT


class MakefileAdversarialTests(OperatorFixture):
    def test_sensitive_proxy_and_cancellation_do_not_replace_history(self):
        endpoint = 'socks5h://127.0.0.1:1080'
        result = self.run_choose('PROXY_URL', endpoint)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        history = self.directory / 'repo/tmp/inputs/PROXY_URL'
        for sensitive in ('http://user:password@proxy:8080', 'https://proxy:8080?token=secret', 'https://proxy:8080#secret'):
            result = self.run_choose('PROXY_URL', sensitive)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((self.directory / 'selected').read_text(), sensitive + '\n')
            self.assertEqual(history.read_text(), endpoint + '\n')
        result = self.run_choose('PROXY_URL', '')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(history.read_text(), endpoint + '\n')

    def test_ambient_context_does_not_select_a_noninteractive_operation(self):
        result = self.run_make('check', DOCKER_CONTEXT='ambient-context')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('a terminal is required', result.stderr)
        self.assertEqual(self.calls(), [['context', 'ls', '--format', '{{.Name}}']])
        self.assertFalse(self.python_log.exists())

    def test_remote_context_is_rejected_before_runtime_command(self):
        result = self.run_make('check', 'DOCKER_CONTEXT=remote', FAKE_DOCKER_ENDPOINT='tcp://remote.example:2376')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must use a local Unix socket', result.stderr)
        self.assertFalse(self.python_log.exists())

    def test_literal_profile_and_shell_sensitive_pool_are_not_evaluated(self):
        pool = '/tmp/cache with $HOME $(touch SHOULD_NOT_RUN)'
        result = self.run_make(
            'build', 'PROFILE=worker-cpu', 'DOCKER_CONTEXT=local', f'WHEELHOUSE={pool}', cwd='/tmp',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.python_arguments()
        self.assertEqual(arguments[1], str(ROOT / 'make/build.py'))
        self.assertEqual(arguments[2:5], ['build', '--profile', 'worker-cpu'])
        self.assertEqual(arguments[5:7], ['--host', 'unix:///tmp/fake-docker.sock'])
        self.assertEqual(arguments[-2:], ['--pool', pool])
        self.assertFalse((self.directory / 'SHOULD_NOT_RUN').exists())
