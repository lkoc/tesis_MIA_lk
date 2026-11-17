async function secuenciaAuditoria(page, collector = "http://127.0.0.1:8766", maximum = 6) {
  async function local(path, payload) {
    return await page.evaluate(async ({url, payload}) => {
      const options = payload === undefined ? {} : {
        method: "POST",
        headers: {"Content-Type": "application/json", "X-Audit-Local": "1"},
        body: JSON.stringify(payload),
      };
      const response = await fetch(url, options);
      const result = await response.json();
      if (!response.ok) throw new Error(JSON.stringify(result));
      return result;
    }, {url: collector + path, payload});
  }

  const completed = [];
  try {
    for (let index = 0; index < maximum; index++) {
      const job = await local("/next");
      if (job.done) return {done: true, completed};
      if (!job.ticket && !job.id_informe_reutilizado) {
        await page.goto("https://sloptotal.com/engines/");
        await page.locator("textarea").waitFor({state: "visible", timeout: 30000});
        await page.locator("textarea").fill(job.texto_formulario);
        const digest = await page.locator("textarea").evaluate(async element => {
          const bytes = new TextEncoder().encode(element.value.replace(/\r\n/g, "\n").trim());
          return Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
            .map(value => value.toString(16).padStart(2, "0")).join("");
        });
        if (digest !== job.sha256_formulario_esperado) throw new Error("Texto distinto: " + job.id);
        await local("/event", {action: "started", id: job.id, sha256: digest});
        await page.getByRole("button", {name: "Analyze text", exact: true}).click();
        await page.waitForURL(/\/report\/\?(ticket|id)=/, {timeout: 30000});
        const action = await page.evaluate(() =>
          new URL(location.href).searchParams.has("ticket") ? "submitted" : "reused");
        await local("/event", {action, id: job.id, url: page.url()});
      } else if (page.url() !== job.url_ticket && page.url() !== job.url_informe) {
        await page.goto(job.url_informe || job.url_ticket);
      }

      await page.getByRole("button", {name: "Not sure", exact: true})
        .waitFor({state: "visible", timeout: 3600000});
      if (await page.evaluate(() => new URL(location.href).searchParams.has("id"))) {
        await page.reload();
        await page.getByRole("button", {name: "Not sure", exact: true})
          .waitFor({state: "visible", timeout: 120000});
      }
      await page.evaluate(() => {
        const original = URL.createObjectURL;
        window.__auditOriginalExport = null;
        URL.createObjectURL = function(blob) {
          blob.text().then(text => { window.__auditOriginalExport = text; });
          return original.call(URL, blob);
        };
      });
      await page.getByRole("button", {name: "Download JSON", exact: true}).click();
      await page.waitForFunction(() => window.__auditOriginalExport !== null, {timeout: 10000});
      const capture = await page.evaluate(() => {
        const reportText = document.querySelector("main").innerText;
        const match = reportText.match(/(?:^|\n)(\d+(?:\.\d+)?)\s*\/\s*100/);
        return {
          captured_at: new Date().toISOString(),
          url: location.href,
          report_id: new URL(location.href).searchParams.get("id"),
          report_text: reportText,
          visible_score: match ? Number(match[1]) : null,
          visible_engine_rows: Array.from(document.querySelectorAll("tbody tr"))
            .map(row => row.innerText),
          export_text: window.__auditOriginalExport,
        };
      });
      const receipt = await local("/event", {action: "capture", id: job.id, capture});
      completed.push({id: job.id, url: capture.url, score: capture.visible_score, receipt});
    }
    return {done: (await local("/next")).done === true, completed};
  } catch (error) {
    return {done: false, completed, error: String(error), url: page.url(),
      note: "No se reintenta ningun envio. Revisar manifest.json antes de reanudar."};
  }
}