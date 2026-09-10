// Uses synthetic HTML fixtures only. Never sends requests to the real review server.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

async function main() {
  const browser = await chromium.launch({ headless: true, channel: process.env.BROWSER_CHANNEL || 'chrome' });
  const root = process.argv[2];
  let passed = 0;
  async function scenario(name, options, check) {
    const page = await browser.newPage();
    const errors = [];
    const counts = {};
    const state = { finish: 'idle', batch: 'idle', complete: false, busy: false, ...options };
    page.on('pageerror', error => errors.push(error.message));
    await page.route('http://review.test/**', async route => {
      const request = route.request();
      const endpoint = new URL(request.url()).pathname;
      const key = request.method() + ' ' + endpoint;
      counts[key] = (counts[key] || 0) + 1;
      const json = (body, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
      if (endpoint === '/') return route.fulfill({ contentType: 'text/html', body: fs.readFileSync(path.join(root, state.pending ? 'pending.html' : 'empty.html'), 'utf8') });
      if (endpoint === '/finish-status') {
        if (state.failFinishRead) { state.failFinishRead = false; return route.abort(); }
        return json({ status: state.finish, step: state.finish, message: 'Review saved', report: 'test report' });
      }
      if (endpoint === '/batch-status') {
        if (state.delayBatchRead && counts[key] > 1) await new Promise(resolve => setTimeout(resolve, 400));
        return json({ status: state.batch, complete: state.complete, step: state.complete ? 'Queue complete' : state.batch });
      }
      if (endpoint === '/next-batch') {
        if (state.conflict) {
          if (state.conflict === 'finishing') state.finish = 'running';
          if (state.conflict === 'review_finished') return json({ reason: state.conflict, status: 'completed', complete: true }, 202);
          return json({ error: 'busy', reason: state.conflict }, 409);
        }
        if (state.networkError) { state.networkError = false; return route.abort(); }
        state.batch = 'completed'; state.complete = true;
        return json({ status: 'completed', complete: true }, 202);
      }
      if (endpoint === '/jobs') {
        const snapshot = { jobs: state.jobs || [], resolved_item_keys: state.resolved ? ['fixture'] : [], summary: { queued: state.queued ?? (state.busy ? 1 : 0), running: 0, failed: state.failed || 0 } };
        if (state.delayJobs) await new Promise(resolve => setTimeout(resolve, state.delayJobs));
        if (state.failJobsRead) { state.failJobsRead = false; return route.abort(); }
        return json(snapshot);
      }
      if (endpoint === '/progress') return json({ progress: { reviewed: 5, total: 5 } });
      if (endpoint === '/finish') {
        state.finish = state.finishFailure ? 'failed' : state.finishRunning ? 'running' : 'completed';
        if (state.lostFinishResponse) { state.lostFinishResponse = false; return route.abort(); }
        return json({ status: state.finish, message: 'saved', report: 'test' }, 202);
      }
      if (endpoint === '/skip') { state.resolved = true; return json({ skipped: 1 }); }
      if (endpoint === '/face' || endpoint === '/image') return route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40"><rect width="40" height="40" fill="#777"/></svg>' });
      throw Error('Unexpected endpoint: ' + endpoint);
    });
    try {
      await page.goto('http://review.test/');
      await page.waitForTimeout(150);
      const finish = async () => {
        await page.locator('#finishReview').click();
        await page.locator('#confirmDialog button[value="confirm"]').click();
      };
      await check(page, state, counts, finish);
      assert.deepEqual(errors, [], name);
      console.log('PASS ' + name);
      passed++;
    } finally { await page.close(); }
  }
  const batchPosts = counts => counts['POST /next-batch'] || 0;
  const quiet = async page => page.waitForTimeout(1600);
  const finished = page => page.waitForFunction(() => document.querySelector('#finishReview').textContent === 'Review Finished');
  try {
    await scenario('finished reload never requests batches', { finish: 'completed' }, async (page, state, counts) => {
      await quiet(page); assert.equal(batchPosts(counts), 0);
      assert.equal(await page.locator('#loadNextBatch').isDisabled(), true);
      await page.reload(); await quiet(page); assert.equal(batchPosts(counts), 0);
    });
    await scenario('empty queue stops after one request and finish is terminal', {}, async (page, state, counts, finish) => {
      await quiet(page); assert.equal(batchPosts(counts), 1);
      await finish(); await finished(page); await page.evaluate(() => { showCluster(); pollJobs(); maybeLoadNextBatch(true); });
      await quiet(page); assert.equal(batchPosts(counts), 1); assert.equal(counts['POST /finish'], 1);
    });
    await scenario('completed queue is remembered across reloads', { batch: 'completed', complete: true }, async (page, state, counts) => {
      await quiet(page); assert.equal(batchPosts(counts), 0);
      await page.locator('#loadNextBatch').click(); await quiet(page); assert.equal(batchPosts(counts), 1);
      await page.reload(); await quiet(page); assert.equal(batchPosts(counts), 1);
    });
    await scenario('other tab finishing is polled without batch retries', { conflict: 'finishing' }, async (page, state, counts) => {
      await quiet(page); assert.equal(batchPosts(counts), 1);
      state.finish = 'completed'; await finished(page); await quiet(page); assert.equal(batchPosts(counts), 1);
    });
    await scenario('other tab already finished returns terminal batch response', { conflict: 'review_finished' }, async (page, state, counts) => {
      await finished(page); await quiet(page); assert.equal(batchPosts(counts), 1);
    });
    await scenario('wait for actions then resume exactly once', { conflict: 'actions_pending', busy: true }, async (page, state, counts) => {
      await quiet(page); assert.equal(batchPosts(counts), 1);
      state.conflict = null; state.busy = false;
      await quiet(page); assert.equal(batchPosts(counts), 2);
      await quiet(page); assert.equal(batchPosts(counts), 2);
    });
    await scenario('finish while waiting for actions cancels retry', { conflict: 'actions_pending', busy: true }, async (page, state, counts, finish) => {
      await finish(); await finished(page); state.busy = false;
      await quiet(page); assert.equal(batchPosts(counts), 1);
    });
    await scenario('finish drains thirteen actions without resuming batches', { batch: 'completed', complete: true, queued: 13, finishRunning: true }, async (page, state, counts, finish) => {
      await page.waitForFunction(() => document.querySelector('#queueCount').textContent === '13');
      await page.evaluate(() => { activeJobIds.add('last-action'); persistJobs(); });
      await finish();
      state.queued = 0;
      state.jobs = [{ id: 'last-action', status: 'completed', item_keys: [], message: 'Saved' }];
      await page.waitForFunction(() => document.querySelector('#queueCount').textContent === '0');
      assert.equal(await page.evaluate(() => sessionStorage.getItem('unknownJobs')), '[]');
      assert.equal(counts['GET /progress'] || 0, 0);
      assert.equal(batchPosts(counts), 0);
      assert.equal(counts['GET /'], 1);
      state.finish = 'completed'; await finished(page); await quiet(page);
      const reads = counts['GET /jobs'];
      await quiet(page); assert.equal(counts['GET /jobs'], reads);
      assert.equal(counts['POST /finish'], 1); assert.equal(batchPosts(counts), 0);
    });
    await scenario('late action response reconciles after finish starts', { batch: 'completed', complete: true, queued: 13, finishRunning: true }, async (page, state, counts, finish) => {
      await page.waitForFunction(() => document.querySelector('#queueCount').textContent === '13');
      state.delayJobs = 400;
      await page.evaluate(() => { pollJobs(); });
      await finish(); state.queued = 0;
      await page.waitForFunction(() => document.querySelector('#queueCount').textContent === '0');
      assert.equal(counts['GET /progress'] || 0, 0); assert.equal(batchPosts(counts), 0);
    });
    await scenario('terminal finish rereads queue after an older in-flight snapshot', { batch: 'completed', complete: true, queued: 13 }, async (page, state, counts, finish) => {
      await page.waitForFunction(() => document.querySelector('#queueCount').textContent === '13' && !polling);
      state.delayJobs = 800;
      const reads = counts['GET /jobs'];
      await page.evaluate(() => { pollJobs(); });
      while (counts['GET /jobs'] === reads) await page.waitForTimeout(10);
      state.queued = 0; state.delayJobs = 0;
      await finish(); await finished(page);
      await page.waitForFunction(() => document.querySelector('#queueCount').textContent === '0' && !polling);
      await quiet(page); assert.equal(batchPosts(counts), 0);
      const finalReads = counts['GET /jobs']; await quiet(page);
      assert.equal(counts['GET /jobs'], finalReads);
    });
    await scenario('finishing reload recovers queue and transient read failure', { finish: 'running', queued: 13 }, async (page, state, counts) => {
      await page.waitForFunction(() => document.querySelector('#queueCount').textContent === '13');
      await page.reload();
      await page.waitForFunction(() => document.querySelector('#queueCount').textContent === '13');
      state.failJobsRead = true;
      await page.waitForFunction(() => document.querySelector('#toast').textContent.includes('fetch'));
      assert.equal(await page.locator('#queueCount').textContent(), '13');
      state.queued = 0;
      await page.waitForFunction(() => document.querySelector('#queueCount').textContent === '0');
      assert.equal(batchPosts(counts), 0);
    });
    await scenario('failed action during finish is reported without reload', { batch: 'completed', complete: true, queued: 1, finishRunning: true }, async (page, state, counts, finish) => {
      await page.waitForFunction(() => document.querySelector('#queueCount').textContent === '1');
      await finish(); state.queued = 0; state.failed = 1;
      state.jobs = [{ id: 'failure', status: 'failed', item_keys: [], error: 'Test action failed safely' }];
      await page.waitForFunction(() => document.querySelector('#toast').textContent === 'Test action failed safely');
      await quiet(page);
      assert.equal(await page.locator('#queueCount').textContent(), '0');
      assert.equal(counts['GET /'], 1); assert.equal(batchPosts(counts), 0);
    });
    await scenario('unknown conflict pauses instead of retry storm', { conflict: 'unexpected' }, async (page, state, counts) => {
      await quiet(page); assert.equal(batchPosts(counts), 1);
      await page.evaluate(() => { showCluster(); pollJobs(); }); await quiet(page); assert.equal(batchPosts(counts), 1);
    });
    await scenario('network failure requires deliberate retry', { networkError: true }, async (page, state, counts) => {
      await quiet(page); assert.equal(batchPosts(counts), 1);
      await page.locator('#loadNextBatch').click(); await quiet(page); assert.equal(batchPosts(counts), 2);
    });
    await scenario('failed finish can be retried without loading batches', { batch: 'completed', complete: true, finishFailure: true }, async (page, state, counts, finish) => {
      await finish(); await page.waitForFunction(() => document.querySelector('#finishReview').textContent === 'Retry Finish Review');
      state.finishFailure = false; await finish(); await finished(page); await quiet(page);
      assert.equal(counts['POST /finish'], 2); assert.equal(batchPosts(counts), 0);
    });
    await scenario('lost finish response reconciles with server', { batch: 'completed', complete: true, lostFinishResponse: true }, async (page, state, counts, finish) => {
      await finish(); await finished(page); await quiet(page); assert.equal(counts['POST /finish'], 1); assert.equal(batchPosts(counts), 0);
    });
    await scenario('filters never abandon unresolved clusters', { pending: true }, async (page, state, counts) => {
      await page.locator('#search').fill('no such person'); await quiet(page); assert.equal(batchPosts(counts), 0);
      await page.locator('#search').fill(''); await page.locator('#skipCluster').click(); await quiet(page);
      assert.equal(batchPosts(counts), 1);
    });
    await scenario('late batch response cannot overwrite finished state', { batch: 'running', delayBatchRead: true }, async (page, state, counts) => {
      await page.evaluate(() => { finishState({status:'completed'}); });
      await quiet(page); await finished(page); assert.equal(batchPosts(counts), 0);
      assert.equal(await page.locator('#clusterPosition').textContent(), 'Review Finished');
    });
    await scenario('unavailable lifecycle fails closed', { failFinishRead: true }, async (page, state, counts) => {
      await quiet(page); assert.equal(batchPosts(counts), 0);
      assert.equal(await page.locator('#loadNextBatch').isDisabled(), true);
    });
    await scenario('pending layout fits desktop and mobile', { pending: true }, async (page, state, counts) => {
      for (const [width, height] of [[1440, 1000], [390, 844]]) {
        await page.setViewportSize({width, height});
        await page.screenshot({path: path.join(process.env.REVIEW_SCREENSHOT_DIR || root, `review-lifecycle-${width}.png`), fullPage: true});
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      }
      assert.equal(batchPosts(counts), 0);
    });
    console.log(`${passed} browser regression scenarios passed`);
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
