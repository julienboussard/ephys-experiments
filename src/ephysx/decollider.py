import torch
from torch import nn

# TODO implement WaveformDenoiser versions


class Decollider(nn.Module):

    def predict(
        self, noisy_waveforms, noisier_waveforms=None, channel_masks=None, n2n_alpha=1.0
    ):
        # multi-chan prediction
        # multi-chan nets below naturally implement this in their
        # forward(), but single-chan nets need a little logic
        return self.forward(noisy_waveforms, channel_masks=channel_masks)

    model_predict = predict

    def loss(
        self, criterion, noisy_waveforms, noisier_waveforms=None, channel_masks=None
    ):
        pred = self.predict(
            noisy_waveforms=noisy_waveforms,
            noisier_waveforms=noisier_waveforms,
            channel_masks=channel_masks,
        )

        if channel_masks is not None:
            channel_masks = channel_masks.to(pred.dtype)
            loss = criterion(
                pred * channel_masks[:, :, None],
                noisy_waveforms * channel_masks[:, :, None],
            )
        else:
            loss = criterion(pred, noisy_waveforms)

        return loss


class Noisier2NoiseMixin:

    def model_predict(
        self, noisy_waveforms, noisier_waveforms, channel_masks=None, n2n_alpha=1.0
    ):
        """See Noisier2Noise paper. This is their Eq. 6.

        If you plan to use this at inference time, then multiply your noise2
        during training by n2n_alpha.
        """
        expected_noisy_waveforms = self.predict(
            noisier_waveforms, channel_masks=channel_masks
        )
        if n2n_alpha == 1.0:
            return 2.0 * expected_noisy_waveforms - noisier_waveforms
        a2inv = 1.0 / (n2n_alpha * n2n_alpha)
        a2p1 = 1.0 + n2n_alpha * n2n_alpha
        return a2inv * (a2p1 * expected_noisy_waveforms - noisier_waveforms)


# -- single channel decolliders


class SingleChannelPredictor(Decollider):
    def predict(self, waveforms, channel_masks=None):
        """NCT -> NCT"""
        n, c, t = waveforms.shape
        waveforms = waveforms.reshape(n * c, 1, t)
        preds = self.forward(waveforms)
        return preds.reshape(n, c, t)


class SingleChannelDecollider(Noisier2NoiseMixin, SingleChannelPredictor):
    def forward(self, waveforms, channel_masks=None):
        """N1T -> N1T"""
        return self.net(waveforms)


class ConvToLinearSingleChannelDecollider(SingleChannelDecollider):
    def __init__(
        self,
        out_channels=(16, 32, 64),
        kernel_lengths=(5, 7, 11),
        hidden_linear_dims=(),
        spike_length_samples=121,
        final_activation="relu",
    ):
        super().__init__()
        assert len(out_channels) == len(kernel_lengths)
        in_channels = (1, *out_channels[:-1])
        is_hidden = [True] * (len(out_channels) - 1) + [False]
        self.net = nn.Sequential()
        for ic, oc, k, hid in zip(in_channels, out_channels, kernel_lengths, is_hidden):
            self.net.append(nn.Conv1d(ic, oc, k))
            if hid:
                self.net.append(nn.ReLU())
        self.net.append(nn.Flatten())
        flat_dim = out_channels[-1] * (
            spike_length_samples - sum(kernel_lengths) + len(kernel_lengths)
        )

        lin_in_dims = (flat_dim,) + hidden_linear_dims
        lin_out_dims = hidden_linear_dims + (spike_length_samples,)
        is_final = [False] * len(hidden_linear_dims) + [True]
        for inf, outf, fin in zip(lin_in_dims, lin_out_dims, is_final):
            if not fin:
                self.net.append(nn.ReLU())
            elif final_activation == "sigmoid":
                self.net.append(nn.Sigmoid())
            elif final_activation == "tanh":
                self.net.append(nn.Tanh())
            elif final_activation == "relu":
                self.net.append(nn.ReLU())
            else:
                assert False
            self.net.append(nn.Linear(inf, outf))
        # add the empty channel dim back in
        self.net.append(nn.Unflatten(1, (1, spike_length_samples)))
        self._kwargs = dict(
            out_channels=out_channels,
            kernel_lengths=kernel_lengths,
            hidden_linear_dims=hidden_linear_dims,
            spike_length_samples=spike_length_samples,
            final_activation=final_activation,
        )


class MLPSingleChannelDecollider(SingleChannelDecollider):
    def __init__(
        self,
        hidden_sizes=(512, 256, 256),
        spike_length_samples=121,
        final_activation="relu",
    ):
        super().__init__()
        self.net = nn.Sequential()
        self.net.append(nn.Flatten())
        input_sizes = (spike_length_samples, *hidden_sizes[:-1])
        output_sizes = hidden_sizes
        is_final = [False] * max(0, len(hidden_sizes) - 1) + [True]
        for inf, outf, fin in zip(input_sizes, output_sizes, is_final):
            self.net.append(nn.Linear(inf, outf))
            if not fin:
                self.net.append(nn.ReLU())
            elif final_activation == "sigmoid":
                self.net.append(nn.Sigmoid())
            elif final_activation == "tanh":
                self.net.append(nn.Tanh())
            elif final_activation == "relu":
                self.net.append(nn.ReLU())
            else:
                assert False
        self.net.append(nn.Linear(hidden_sizes[-1], spike_length_samples))
        # add the empty channel dim back in
        self.net.append(nn.Unflatten(1, (1, spike_length_samples)))
        self._kwargs = dict(
            hidden_sizes=hidden_sizes,
            spike_length_samples=spike_length_samples,
            final_activation=final_activation,
        )


# -- multi channel decolliders


class MultiChannelDecollider(Noisier2NoiseMixin, Decollider):
    """NCT -> NCT

    self.net must map NC2T -> NCT

    Mask is added like so:
    waveforms  NCT  ->  N1CT \
    masks      NC   ->  N1C1 -> N2CT (broadcast and concat)
    """

    def forward(self, waveforms, channel_masks=None):
        # add the masks as an input channel
        # I somehow feel that receiving a "badness indicator" is more useful,
        # and the masks indicate good channels, so hence the flip below
        if channel_masks is None:
            masks = torch.zeros_like(waveforms[:, :, 0])
        else:
            assert channel_masks.shape == waveforms.shape[:2]
            masks = torch.logical_not(channel_masks).to(waveforms)
        # NCT -> N1CT (channels are height in Conv2D NCHW convention)
        waveforms = waveforms[:, None, :, :]
        # NC -> N1CT
        masks = torch.broadcast_to(masks[:, None, :, None], waveforms.shape)
        # -> N2CT, concatenate on channel dimension (NCHW)
        combined = torch.concatenate((waveforms, masks), dim=1)
        return self.net(combined)


class ConvToLinearMultiChannelDecollider(MultiChannelDecollider):
    def __init__(
        self,
        out_channels=(16, 32),
        kernel_heights=(4, 4),
        kernel_lengths=(5, 5),
        hidden_linear_dims=(1024,),
        n_channels=1,
        spike_length_samples=121,
        final_activation="relu",
    ):
        super().__init__()
        assert len(out_channels) == len(kernel_heights) == len(kernel_lengths)
        in_channels = (2, *out_channels[:-1])
        is_hidden = [True] * (len(out_channels) - 1) + [False]
        self.net = nn.Sequential()

        for ic, oc, kl, kh, hid in zip(
            in_channels, out_channels, kernel_lengths, kernel_heights, is_hidden
        ):
            self.net.append(nn.Conv2d(ic, oc, (kh, kl)))
            if hid:
                self.net.append(nn.ReLU())
        self.net.append(nn.Flatten())
        out_w = spike_length_samples - sum(kernel_lengths) + len(kernel_lengths)
        out_h = n_channels - sum(kernel_heights) + len(kernel_heights)
        flat_dim = out_channels[-1] * out_w * out_h
        lin_in_dims = (flat_dim,) + hidden_linear_dims
        lin_out_dims = hidden_linear_dims + (n_channels * spike_length_samples,)
        is_final = [False] * len(hidden_linear_dims) + [True]
        for inf, outf, fin in zip(lin_in_dims, lin_out_dims, is_final):
            if not fin:
                self.net.append(nn.ReLU())
            elif final_activation == "sigmoid":
                self.net.append(nn.Sigmoid())
            elif final_activation == "tanh":
                self.net.append(nn.Tanh())
            elif final_activation == "relu":
                self.net.append(nn.ReLU())
            else:
                assert False
            self.net.append(nn.Linear(inf, outf))
        self.net.append(nn.Unflatten(1, (n_channels, spike_length_samples)))
        self._kwargs = dict(
            out_channels=out_channels,
            kernel_heights=kernel_heights,
            kernel_lengths=kernel_lengths,
            hidden_linear_dims=hidden_linear_dims,
            n_channels=n_channels,
            spike_length_samples=spike_length_samples,
            final_activation=final_activation,
        )


class MLPMultiChannelDecollider(MultiChannelDecollider):
    def __init__(
        self,
        hidden_sizes=(1024, 512, 512),
        n_channels=1,
        spike_length_samples=121,
        final_activation="relu",
    ):
        super().__init__()
        self.net = nn.Sequential()
        self.net.append(nn.Flatten())
        input_sizes = (
            2 * n_channels * spike_length_samples,
            *hidden_sizes[:-1],
        )
        output_sizes = hidden_sizes
        is_final = [False] * max(0, len(hidden_sizes) - 1) + [True]
        for inf, outf, fin in zip(input_sizes, output_sizes, is_final):
            self.net.append(nn.Linear(inf, outf))
            if not fin:
                self.net.append(nn.ReLU())
            elif final_activation == "sigmoid":
                self.net.append(nn.Sigmoid())
            elif final_activation == "tanh":
                self.net.append(nn.Tanh())
            elif final_activation == "relu":
                self.net.append(nn.ReLU())
            else:
                assert False
        self.net.append(nn.Linear(hidden_sizes[-1], n_channels * spike_length_samples))
        self.net.append(nn.Unflatten(1, (n_channels, spike_length_samples)))
        self._kwargs = dict(
            hidden_sizes=hidden_sizes,
            n_channels=n_channels,
            spike_length_samples=spike_length_samples,
            final_activation=final_activation,
        )


# -- idea which i am jokingly calling Noisier3Noise for now


class Noisier3Noise(Decollider):
    def __init__(self, waveform_net, noise_net):
        super().__init__()
        self.waveform_net = waveform_net
        self.noise_net = noise_net

    def predict(
        self, noisy_waveforms, noisier_waveforms=None, channel_masks=None, n2n_alpha=1.0
    ):
        return self.waveform_net.predict(
            noisy_waveforms,
            noisier_waveforms=noisier_waveforms,
            channel_masks=channel_masks,
            n2n_alpha=n2n_alpha,
        )

    def model_predict(
        self, noisy_waveforms, noisier_waveforms=None, channel_masks=None, n2n_alpha=1.0
    ):
        wf_pred = self.waveform_net.predict(
            noisier_waveforms,
            noisier_waveforms=noisier_waveforms,
            channel_masks=channel_masks,
            n2n_alpha=n2n_alpha,
        )
        noise_pred = self.noise_net.predict(
            noisier_waveforms,
            noisier_waveforms=noisier_waveforms,
            channel_masks=channel_masks,
            n2n_alpha=n2n_alpha,
        )
        return wf_pred - noise_pred

    def loss(
        self, criterion, noisy_waveforms, noisier_waveforms=None, channel_masks=None
    ):
        wf_loss = self.waveform_net.loss(
            criterion, noisy_waveforms, noisier_waveforms=noisier_waveforms, channel_masks=channel_masks
        )
        noise2 = noisier_waveforms - noisy_waveforms
        noise2_loss = self.noise_net.loss(
            criterion, noise2, noisier_waveforms=noisier_waveforms, channel_masks=channel_masks
        )
        loss = wf_loss + noise2_loss
        return loss
