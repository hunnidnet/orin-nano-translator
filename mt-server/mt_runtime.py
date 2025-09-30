import ctranslate2 as ct2
import sentencepiece as spm

class MarianCT2:
    def __init__(self, en2es_dir, es2en_dir, en2es_sp_dir, es2en_sp_dir, device="cuda", compute_type="float16"):
        self.sp_en2es_src = spm.SentencePieceProcessor(model_file=f"{en2es_sp_dir}/source.spm")
        self.sp_en2es_tgt = spm.SentencePieceProcessor(model_file=f"{en2es_sp_dir}/target.spm")
        self.sp_es2en_src = spm.SentencePieceProcessor(model_file=f"{es2en_sp_dir}/source.spm")
        self.sp_es2en_tgt = spm.SentencePieceProcessor(model_file=f"{es2en_sp_dir}/target.spm")

        self.en2es = ct2.Translator(en2es_dir, device=device, compute_type=compute_type)
        self.es2en = ct2.Translator(es2en_dir, device=device, compute_type=compute_type)

    def _prep(self, sp_src, text, tgt_tag):
        return [tgt_tag] + sp_src.encode(text, out_type=str) + ["</s>"]

    def _decode(self, sp_tgt, toks):
        if toks and toks[-1] == "</s>":
            toks = toks[:-1]
        return sp_tgt.decode(toks)

    def translate(self, text, src_lang, tgt_lang):
        s = src_lang.lower(); t = tgt_lang.lower()

        if s.startswith("en") and t.startswith("es"):
            src_tokens = self._prep(self.sp_en2es_src, text, ">>es<<")
            out = self.en2es.translate_batch(
                [src_tokens],
                beam_size=4, length_penalty=1.1, no_repeat_ngram_size=3,
                max_decoding_length=256, end_token="</s>",
            )[0].hypotheses[0]
            return self._decode(self.sp_en2es_tgt, out)

        if s.startswith("es") and t.startswith("en"):
            src_tokens = self._prep(self.sp_es2en_src, text, ">>en<<")
            out = self.es2en.translate_batch(
                [src_tokens],
                beam_size=4, length_penalty=1.1, no_repeat_ngram_size=3,
                max_decoding_length=256, end_token="</s>",
            )[0].hypotheses[0]
            return self._decode(self.sp_es2en_tgt, out)

        return text
