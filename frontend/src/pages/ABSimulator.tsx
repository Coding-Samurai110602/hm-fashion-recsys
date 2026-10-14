import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { calcPower, runSimulation } from '../api/endpoints'
import { ErrorState } from '../components/ErrorState'
import { LoadingSpinner } from '../components/LoadingSpinner'

function SliderInput({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <label className="text-gray-700">{label}</label>
        <span className="font-mono text-indigo-600">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-indigo-600"
        aria-label={label}
      />
      <div className="flex justify-between text-xs text-gray-400">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  )
}

export default function ABSimulator() {
  // Power calculator state
  const [baselineRate, setBaselineRate] = useState(0.0857)
  const [relativeLift, setRelativeLift] = useState(1.1)
  const [alpha, setAlpha] = useState(0.05)
  const [power, setPower] = useState(0.8)
  const [weeklyTraffic, setWeeklyTraffic] = useState<string>('')
  const [cupedVarianceReduction] = useState(0.026)

  // Simulation state
  const [testType, setTestType] = useState<'aa' | 'ab'>('ab')
  const [nPerArm, setNPerArm] = useState(500)
  const [nSims, setNSims] = useState(200)
  const [peeking, setPeeking] = useState(false)
  const [nDays, setNDays] = useState(14)

  const powerMutation = useMutation({
    mutationFn: () =>
      calcPower({
        baseline_rate: baselineRate,
        relative_lift: relativeLift,
        alpha,
        power,
        weekly_traffic: weeklyTraffic ? parseInt(weeklyTraffic) : undefined,
        cuped_variance_reduction: cupedVarianceReduction,
      }),
  })

  const simMutation = useMutation({
    mutationFn: () =>
      runSimulation({
        test_type: testType,
        n_per_arm: nPerArm,
        n_sims: nSims,
        peeking,
        n_days: nDays,
        seed: 42,
      }),
  })

  const powerResult = powerMutation.data
  const simResult = simMutation.data

  // Peeking FPR warning
  const showPeekingWarning =
    simResult?.peeking_fpr != null && simResult.false_positive_rate != null

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">A/B Simulator</h1>
        <p className="mt-1 text-gray-500">
          Power calculation and offline permutation simulation using holdout customer data.
        </p>
      </div>

      <div className="rounded-lg border border-yellow-200 bg-yellow-50 px-4 py-3 text-sm text-yellow-800">
        These simulations use offline holdout customer data replayed as a
        permutation test — online traffic, engagement effects, and novelty are
        not modelled.
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        {/* Power calculator */}
        <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm space-y-5">
          <h2 className="text-base font-semibold text-gray-800">
            Power Calculator
          </h2>

          <SliderInput
            label="Baseline hit rate"
            value={baselineRate}
            min={0.01}
            max={0.5}
            step={0.001}
            onChange={setBaselineRate}
          />
          <SliderInput
            label="Relative lift"
            value={relativeLift}
            min={1.01}
            max={2.0}
            step={0.01}
            onChange={setRelativeLift}
          />
          <SliderInput
            label="Alpha (significance)"
            value={alpha}
            min={0.01}
            max={0.2}
            step={0.01}
            onChange={setAlpha}
          />
          <SliderInput
            label="Power"
            value={power}
            min={0.5}
            max={0.99}
            step={0.01}
            onChange={setPower}
          />

          <div className="space-y-1">
            <label className="text-sm text-gray-700">
              Weekly traffic (optional)
            </label>
            <input
              type="number"
              value={weeklyTraffic}
              onChange={(e) => setWeeklyTraffic(e.target.value)}
              placeholder="e.g. 10000"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>

          <div className="rounded-lg bg-gray-50 px-3 py-2 text-xs text-gray-500">
            CUPED variance reduction: {(cupedVarianceReduction * 100).toFixed(1)}%
            (covariate: 4-week purchase count)
          </div>

          <button
            onClick={() => powerMutation.mutate()}
            disabled={powerMutation.isPending}
            className="w-full rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-60"
          >
            {powerMutation.isPending ? 'Calculating...' : 'Calculate Power'}
          </button>

          {powerMutation.isPending && (
            <LoadingSpinner label="Calculating..." />
          )}
          {powerMutation.error && (
            <ErrorState
              error={powerMutation.error as Error}
              title="Calculation failed"
            />
          )}

          {powerResult && (
            <div className="rounded-lg border border-indigo-100 bg-indigo-50 p-4 space-y-2 text-sm">
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <p className="text-xs text-gray-500">n per arm</p>
                  <p className="text-xl font-bold text-indigo-700">
                    {powerResult.n_per_arm.toLocaleString()}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">n per arm (CUPED)</p>
                  <p className="text-xl font-bold text-indigo-700">
                    {powerResult.n_per_arm_cuped?.toLocaleString() ?? '—'}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Cohen h</p>
                  <p className="font-mono text-gray-800">
                    {powerResult.cohen_h.toFixed(4)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Weeks needed</p>
                  <p className="font-mono text-gray-800">
                    {powerResult.weeks_needed?.toFixed(2) ?? '—'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Simulation runner */}
        <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm space-y-5">
          <h2 className="text-base font-semibold text-gray-800">
            Simulation Runner
          </h2>

          {/* Test type */}
          <div className="flex gap-3">
            {(['aa', 'ab'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTestType(t)}
                className={`flex-1 rounded-lg border py-2 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                  testType === t
                    ? 'border-indigo-400 bg-indigo-50 text-indigo-700'
                    : 'border-gray-200 bg-white text-gray-600 hover:bg-gray-50'
                }`}
              >
                {t.toUpperCase()} test
              </button>
            ))}
          </div>

          <SliderInput
            label="n per arm"
            value={nPerArm}
            min={100}
            max={5000}
            step={50}
            onChange={setNPerArm}
          />
          <SliderInput
            label="n simulations"
            value={nSims}
            min={100}
            max={1000}
            step={50}
            onChange={setNSims}
          />
          <SliderInput
            label="n days (for peeking)"
            value={nDays}
            min={7}
            max={30}
            step={1}
            onChange={setNDays}
          />

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={peeking}
              onChange={(e) => setPeeking(e.target.checked)}
              className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
            />
            <span className="text-sm text-gray-700">
              Enable peeking simulation
            </span>
          </label>

          <button
            onClick={() => simMutation.mutate()}
            disabled={simMutation.isPending}
            className="w-full rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-60"
          >
            {simMutation.isPending ? 'Running simulation...' : 'Run Simulation'}
          </button>

          {simMutation.isPending && (
            <LoadingSpinner label="Running simulation..." />
          )}
          {simMutation.error && (
            <ErrorState
              error={simMutation.error as Error}
              title="Simulation failed"
            />
          )}

          {simResult && (
            <div className="rounded-lg border border-gray-100 bg-gray-50 p-4 space-y-3 text-sm">
              <div className="grid grid-cols-2 gap-3">
                {simResult.false_positive_rate != null && (
                  <div>
                    <p className="text-xs text-gray-500">False positive rate (A/A)</p>
                    <p className="text-xl font-bold text-gray-800">
                      {(simResult.false_positive_rate * 100).toFixed(1)}%
                    </p>
                  </div>
                )}
                {simResult.empirical_power != null && (
                  <div>
                    <p className="text-xs text-gray-500">Empirical power</p>
                    <p className="text-xl font-bold text-indigo-700">
                      {(simResult.empirical_power * 100).toFixed(1)}%
                    </p>
                  </div>
                )}
                {simResult.peeking_fpr != null && (
                  <div className="col-span-2">
                    <p className="text-xs text-red-600 font-semibold">
                      Peeking FPR
                    </p>
                    <p className="text-xl font-bold text-red-600">
                      {(simResult.peeking_fpr * 100).toFixed(1)}%
                    </p>
                  </div>
                )}
                {simResult.peeking_power != null && (
                  <div className="col-span-2">
                    <p className="text-xs text-orange-600 font-semibold">
                      Peeking power
                    </p>
                    <p className="text-xl font-bold text-orange-600">
                      {(simResult.peeking_power * 100).toFixed(1)}%
                    </p>
                  </div>
                )}
              </div>

              {showPeekingWarning && (
                <div className="rounded-lg border border-red-200 bg-red-50 p-3">
                  <p className="text-sm font-semibold text-red-700">
                    Peeking inflates FPR:{' '}
                    {(simResult.false_positive_rate! * 100).toFixed(1)}% →{' '}
                    {(simResult.peeking_fpr! * 100).toFixed(1)}%
                  </p>
                  <p className="mt-1 text-xs text-red-600">
                    Looking at results before the planned end date inflates the
                    false positive rate significantly.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
