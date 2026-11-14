"""Synthetic CPU tests. No corpus, model download, credentials or GPU required."""
import contextlib
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import soundfile as sf

from algorithms import correct_gradients, proximal_penalty, update_control, weighted_average
from config import Settings
from data import AudioTextDataset
from engine import run
from metrics import choose_threshold, metrics, summarize_scores
from partitions import PARTITIONS, assign_participants, validate_windows
from schedule import epoch_plan, learning_rate, make_units
from storage import seed_all
from windows import create_candidate_starts


def synthetic_manifest():
    # Deliberately synthetic IDs, labels, paths and windows, unrelated to actual participants.
    return pd.DataFrame([dict(participant_id=str(10000+pid), sample_id=f'synthetic_{pid}_{w}',
        label=int(pid < 30), split='train', start_time=float(w), end_time=float(w+30),
        audio_path='synthetic.wav', transcript_path='synthetic.csv')
        for pid in range(107) for w in range(200)])


class PartitionTests(unittest.TestCase):
    def test_all_paper_counts_and_disjointness(self):
        frame = synthetic_manifest()
        for name, expected in PARTITIONS.items():
            clients = assign_participants(frame, name)
            people = set()
            for client, (total, positive) in zip(clients, expected):
                self.assertEqual(client.participant_id.nunique(), total)
                self.assertEqual(client.groupby('participant_id').label.first().sum(), positive)
                self.assertFalse(people & set(client.participant_id))
                people.update(client.participant_id)
            self.assertEqual(sum(map(len, clients)), 21400)
            repeated = assign_participants(frame, name)
            for a, b in zip(clients, repeated):
                pd.testing.assert_frame_equal(a, b)

    def test_duplicate_window_rejected(self):
        frame = synthetic_manifest()
        frame.loc[1, ['start_time', 'end_time']] = frame.loc[0, ['start_time', 'end_time']]
        with self.assertRaises(ValueError):
            validate_windows(frame, 'train')

    def test_no_speech_eval_window_rejected(self):
        transcript = pd.DataFrame(dict(speaker=['Ellie'], start_time=[0.0], stop_time=[100.0]))
        self.assertEqual(len(create_candidate_starts(transcript, 0, 100, 30)), 0)


class InputTests(unittest.TestCase):
    def test_interviewer_role_and_audio_interval_merging(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            waveform = np.linspace(-.5, .5, 48000, dtype=np.float32)
            sf.write(folder/'synthetic.wav', waveform, 16000, subtype='FLOAT')
            pd.DataFrame(dict(start_time=[0., .5, .76, 1.5, 2.],
                stop_time=[.5, .75, 1., 2., 2.5],
                speaker=['Ellie', 'Participant', 'Participant', 'Ellie', 'Participant'],
                value=['question', 'a', 'b', 'follow-up', 'answer'])).to_csv(folder/'synthetic.csv', index=False)
            frame = pd.DataFrame([dict(sample_id='synthetic', participant_id='10000', label=1,
                audio_path='synthetic.wav', transcript_path='synthetic.csv', start_time=0., end_time=3.)])
            item = AudioTextDataset(frame, folder)[0]
            expected = np.concatenate([waveform[8000:16000], waveform[32000:40000]])
            np.testing.assert_array_equal(item['audio'], expected)
            self.assertEqual(item['conversation_text'],
                'Interviewer: question\nParticipant: a\nParticipant: b\nInterviewer: follow-up\nParticipant: answer')
            self.assertEqual(item['target_text'], 'Depressed')


class AlgorithmTests(unittest.TestCase):
    def test_weighted_adapter_average(self):
        result = weighted_average([{'a': torch.tensor([1., 3.])}, {'a': torch.tensor([5., 7.])}], [3, 1])
        torch.testing.assert_close(result['a'], torch.tensor([2., 4.]))
        with self.assertRaises(ValueError):
            weighted_average([{'a': torch.tensor([float('nan')])}], [1])

    def test_proximal_gradient_is_mu_times_displacement(self):
        parameter = torch.nn.Parameter(torch.tensor([2., 3.]))
        proximal_penalty([('a', parameter)], {'a': torch.tensor([1., 1.])}, .01).backward()
        torch.testing.assert_close(parameter.grad, torch.tensor([.01, .02]))

    def test_scaffold_correction_sign_and_control_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        parameter.grad = torch.tensor([2.])
        correct_gradients([('a', parameter)], {'a': torch.tensor([3.])}, {'a': torch.tensor([5.])})
        torch.testing.assert_close(parameter.grad, torch.tensor([4.]))
        state = update_control({'a': torch.tensor([2.])}, {'a': torch.tensor([1.])},
            {'a': torch.tensor([3.])}, {'a': torch.tensor([5.])}, .5)
        torch.testing.assert_close(state['a'], torch.tensor([0.]))

    def test_full_participation_server_control_invariant(self):
        zero = {'a': torch.zeros(2)}
        controls = [zero, zero, zero]
        server = zero
        sizes = [54, 32, 21]
        for _ in range(3):
            controls = [update_control({'a': torch.ones(2)},
                {'a': torch.ones(2)-.01*(cid+1)}, controls[cid], server, .03)
                for cid in range(3)]
            previous_server = server
            server = weighted_average(controls, sizes)
            deltas = [{k: c[k]-old[k] for k in c} for c, old in zip(controls,
                [zero, zero, zero] if _ == 0 else previous_controls)]
            expected_delta = weighted_average(deltas, sizes)
            torch.testing.assert_close(server['a'], previous_server['a']+expected_delta['a'])
            previous_controls = controls


class ScheduleTests(unittest.TestCase):
    def test_all_windows_once_per_epoch_all_partitions(self):
        for counts in PARTITIONS.values():
            sizes = [n*200 for n, _ in counts]
            units = make_units(sizes, Settings())
            self.assertEqual(len(units), 45)
            self.assertEqual(sum(sum(map(len, u['batches'])) for u in units), 64200)
            for epoch in range(3):
                for cid, size in enumerate(sizes):
                    indices = [i for u in units if u['epoch'] == epoch and u['client'] == cid
                               for batch in u['batches'] for i in batch]
                    self.assertEqual(sorted(indices), list(range(size)))
        iid = make_units([7200, 7200, 7000], Settings())
        self.assertEqual(sum(len(u['batches']) for u in iid), 3096)

    def test_learning_rate_warmup_and_decay(self):
        settings = Settings()
        self.assertAlmostEqual(learning_rate(0, 100, settings), 3e-7)
        self.assertAlmostEqual(learning_rate(10, 100, settings), 3e-6)
        self.assertEqual(learning_rate(300, 100, settings), 0.0)


class MetricTests(unittest.TestCase):
    def test_auc_ap_ties(self):
        values = metrics([0, 1, 0, 1], [0., 0., 1., 1.], 0)
        self.assertAlmostEqual(values['roc_auc'], .5)
        self.assertAlmostEqual(values['average_precision'], .5)
        self.assertEqual([values[k] for k in ['tn', 'fp', 'fn', 'tp']], [1, 1, 1, 1])

    def test_perfect_scores_and_no_test_threshold_fitting(self):
        y, score = [0, 0, 1, 1], [-2., -1., 1., 2.]
        threshold = choose_threshold(y, score)
        self.assertEqual(threshold, 0.)
        self.assertEqual(metrics(y, score, threshold)['f1_positive'], 1.)
        with self.assertRaises(ValueError):
            choose_threshold(y, score, 'test')

    def test_participant_mean_and_frozen_threshold(self):
        manifest = pd.DataFrame([dict(sample_id=f's_{pid}_{w}', participant_id=str(pid), label=pid)
                                 for pid in range(2) for w in range(200)])
        scores = manifest.rename(columns={'label': 'true_label'}).copy()
        scores['nll_margin'] = np.repeat([-1., 1.], 200)
        people, result = summarize_scores(scores, manifest, 'dev')
        self.assertEqual(len(people), 2)
        self.assertEqual(result['f1_positive'], 1.)
        _, tested = summarize_scores(scores, manifest, 'test', result['threshold'])
        self.assertEqual(tested['threshold'], result['threshold'])
        with self.assertRaises(ValueError):
            summarize_scores(scores, manifest, 'test')


class TinyBackend:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor([.1, -.2]))

    def parameters(self):
        return [('adapter', self.weight)]

    def state(self):
        return {'adapter': self.weight.detach().clone()}

    def load(self, state):
        with torch.no_grad():
            self.weight.copy_(state['adapter'])

    def train(self, mode=True):
        pass

    def loss(self, item):
        # Stochastic factor exercises RNG recovery as well as optimizer moment recovery.
        return ((self.weight*item[0]-item[1])**2).mean()*(.5+torch.rand(()))


class ResumeTests(unittest.TestCase):
    def test_all_five_methods_match_uninterrupted_after_midsegment_resume(self):
        settings = replace(Settings(), segments=2, batch=21, learning_rate=.001)
        datasets = [[(torch.tensor([1., float(i % 3)]), torch.tensor([.2, -.3]))
                     for i in range(n)] for n in [130, 131, 129]]
        for method in ['central', 'local', 'fedavg', 'fedprox', 'scaffold']:
            selected = datasets[:1] if method == 'central' else datasets
            with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
                root = Path(temp)
                identity = dict(test='synthetic', method=method)
                seed_all(42)
                whole = run(TinyBackend(), selected, method, settings, root/'whole', identity)
                seed_all(42)
                part = run(TinyBackend(), selected, method, settings, root/'resumed', identity, max_new_steps=5)
                self.assertIsNotNone(part['active'])
                seed_all(9999)  # Resume must overwrite this unrelated process RNG.
                resumed = run(TinyBackend(), selected, method, settings, root/'resumed', identity, resume=True)
                self.assertEqual(whole['steps'], resumed['steps'])
                self.assertEqual(whole['windows'], resumed['windows'])
                for key in whole['global_state']:
                    self.assertTrue(torch.equal(whole['global_state'][key], resumed['global_state'][key]), method)
                for a, b in zip(whole['controls'], resumed['controls']):
                    for key in a:
                        self.assertTrue(torch.equal(a[key], b[key]), method)
                if method in ['central', 'local']:
                    for a, b in zip(whole['memory'], resumed['memory']):
                        self.assertTrue(torch.equal(a['adapter']['adapter'], b['adapter']['adapter']), method)
                if method == 'scaffold':
                    expected = weighted_average(resumed['controls'], list(map(len, selected)))
                    self.assertTrue(torch.equal(expected['adapter'], resumed['server_control']['adapter']))
                with self.assertRaises(ValueError):
                    run(TinyBackend(), selected, method, settings, root/'resumed', {'wrong': 1}, resume=True)


if __name__ == '__main__':
    unittest.main()
