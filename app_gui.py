# ==============================================================
# app_gui.py  
# ==============================================================

import contextlib, io, pathlib, sys, threading, traceback
import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
import customtkinter as ctk

# ---------- importa pipeline silenciosamente -------------------
try:
    BASE = pathlib.Path(__file__).resolve().parent
except NameError:
    BASE = pathlib.Path.cwd()

buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    sys.path.append(str(BASE))
    import fuzzynet_pipeline as fn       # v9 já inclui plot_comparative

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class FuzzyApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FUZZYNET-PROT • PPI Predictor")
        self.geometry("900x720")

        # -------- Painel de entrada -------------------------------------
        left = ctk.CTkFrame(self, width=380, corner_radius=10)
        left.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(left, text="Rede:").pack(anchor="w", pady=(10, 0))
        self.model_var = ctk.StringVar(value="ANFIS")
        ctk.CTkOptionMenu(left, values=["ANFIS", "FRNN"],
                          variable=self.model_var)\
           .pack(fill="x", pady=5)

        for lab in ("Seq 1 (FASTA/plain):", "Seq 2:"):
            ctk.CTkLabel(left, text=lab).pack(anchor="w")
            setattr(self, f"seq{1 if '1' in lab else 2}",
                    ctk.CTkTextbox(left, height=90))
            getattr(self, f"seq{1 if '1' in lab else 2}").pack(fill="both",
                                                               pady=5)

        self.k_var = ctk.IntVar(value=3)
        ctk.CTkLabel(left, text="k variáveis:").pack(anchor="w")
        ctk.CTkSlider(left, from_=1, to=10, number_of_steps=9,
                      variable=self.k_var).pack(fill="x", pady=5)

        ctk.CTkButton(left, text="Executar Predição",
                      command=self._start).pack(fill="x", pady=15)

        # -------- Painel de saída --------------------------------------
        right = ctk.CTkFrame(self, corner_radius=10)
        right.pack(expand=True, fill="both", padx=10, pady=10)
        self.status = ctk.CTkLabel(right, text="Pronto.",
                                   font=("Roboto", 15))
        self.status.pack(pady=10)
        self.canvas = tk.Canvas(right, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(expand=True, fill="both")

    # ---------------- execução em thread -------------------------------
    def _start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        s1 = self.seq1.get("1.0", "end").strip()
        s2 = self.seq2.get("1.0", "end").strip()
        if not (s1 and s2):
            return self._set("Forneça as duas sequências.")
        self._set("Processando… aguarde ✨")

        try:
            # -------- predição -----------------------------------------
            cls, pr = fn.predict_interaction(s1, s2, self.model_var.get())
            self._set(f"Classe = {cls}   Prob = {pr.round(3)}")

            model = fn.build_model(self.model_var.get())

            # -------- gráfico global de MFs ---------------------------
            fn.plot_memberships(model, top_k=self.k_var.get(),
                                save_to=BASE / "tmp" / "mf_all.png")

            # -------- vetor completo (2·D) para comparativo -----------
            v1 = fn.reduce_and_scale(fn.get_embedding(s1))
            v2 = fn.reduce_and_scale(fn.get_embedding(s2))
            x_full = np.concatenate([v1, v2]).astype(np.float32)

            # variáveis mais importantes (dimensões) em toda a entrada
            imp = np.abs(model.cons.detach().cpu().numpy()).sum((0, 2))
            top_vars = imp.argsort()[-self.k_var.get():][::-1].tolist()

            fn.plot_comparative(model,
                                torch.tensor(x_full),
                                var_indices=top_vars,
                                save_to=BASE / "tmp" / "mf_comp.png")

            # -------- junta as duas imagens verticalmente -------------
            img_all  = Image.open(BASE / "tmp" / "mf_all.png")
            img_comp = Image.open(BASE / "tmp" / "mf_comp.png")

            new_w = max(img_all.width, img_comp.width)
            new_h = img_all.height + img_comp.height + 10
            merged = Image.new("RGB", (new_w, new_h), color="#1a1a1a")
            merged.paste(img_all,  (0, 0))
            merged.paste(img_comp, (0, img_all.height + 10))

            merged.thumbnail((self.canvas.winfo_width() - 40,
                              self.canvas.winfo_height() - 40))

            self._imgref = ImageTk.PhotoImage(merged)
            self.canvas.delete("all")
            self.canvas.create_image(self.canvas.winfo_width() // 2,
                                     self.canvas.winfo_height() // 2,
                                     image=self._imgref)
        except Exception as e:
            traceback.print_exc()
            self._set(f"Erro: {e}")

    # -------------------------------------------------------------------
    def _set(self, msg: str):
        self.status.configure(text=msg)


if __name__ == "__main__":
    FuzzyApp().mainloop()
