import torch
import torch.nn as nn
import torch.nn.functional as F
from masking import create_mask
from settings import PAD_IDX, BOS_IDX, EOS_IDX

# Implementation from https://github.com/jadore801120/attention-is-all-you-need-pytorch
class Translator(nn.Module):
    """Load a trained model and translate in beam search fashion."""

    def __init__(
        self,
        model,
        beam_size,
        max_seq_len,
    ):

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.trg_bos_idx = BOS_IDX
        self.trg_eos_idx = EOS_IDX

        self.model = model
        self.model.eval()

        self.register_buffer("init_seq", torch.LongTensor([[BOS_IDX]]))
        self.register_buffer(
            "blank_seqs",
            torch.full((beam_size, max_seq_len), PAD_IDX, dtype=torch.long),
        )
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            "len_map", torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0)
        )

    def _model_decode(self, tgt, memory):
        trg_mask, trg_padding_mask = create_mask(tgt)
        trg_mask = trg_mask.to(torch.device("cpu"))
        dec_output = self.model.decode(memory, tgt, trg_mask)
        return F.softmax(self.model.generator(dec_output), dim=-1)

    def _get_init_state(self, src_seq):
        beam_size = self.beam_size

        memory = self.model.encode(src_seq)
        dec_output = self._model_decode(self.init_seq, memory)
        dec_op_permuted = dec_output[-1, :, :]
        best_k_probs, best_k_idx = dec_op_permuted.topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        memory = memory.repeat(1, beam_size, 1)
        return memory, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1

        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        dec_op_permuted = dec_output[-1, :, :]
        best_k2_probs, best_k2_idx = dec_op_permuted.topk(beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(
            beam_size, 1
        )

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = (
            best_k_idx_in_k2 // beam_size,
            best_k_idx_in_k2 % beam_size,
        )
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, src):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src.size(0) == 1

        trg_eos_idx = self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha

        with torch.no_grad():
            memory, gen_seq, scores = self._get_init_state(src)

            ans_idx = 0  # default
            for step in range(2, max_seq_len):  # decode up to max length
                gen_seq_ip = gen_seq[:, :step].transpose(0, 1)
                dec_output = self._model_decode(gen_seq_ip, memory)
                gen_seq, scores = self._get_the_best_score_and_idx(
                    gen_seq, dec_output, scores, step
                )

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][: seq_lens[ans_idx]]
