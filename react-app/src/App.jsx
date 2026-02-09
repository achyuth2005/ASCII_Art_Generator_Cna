import '@mantine/core/styles.css'
import { MantineProvider, createTheme, Container, Title, Text, Textarea, Button, Group, Stack, Paper, Slider, Select, Badge, ActionIcon, Tooltip, Collapse, Box, SimpleGrid, Code, CopyButton, Loader, Switch, NumberInput, Accordion, Tabs, Divider, Center } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { useState } from 'react'
import { Sparkles, Settings, Copy, Check, ChevronDown, Image as ImageIcon, Terminal, Cpu, Palette, Wand2, Moon, Route, Brain, Home, Cat, Star, Mountain, TreePine, Heart, Rocket, Lightbulb, Play } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const theme = createTheme({
  primaryColor: 'blue',
  fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
  defaultRadius: 'md',
  colors: {
    dark: [
      '#C1C2C5', '#A6A7AB', '#909296', '#5C5F66',
      '#373A40', '#2C2E33', '#25262B', '#1A1B1E',
      '#141517', '#101113'
    ],
  },
})

const EXAMPLES = [
  { icon: Home, label: 'House' },
  { icon: Cat, label: 'Cat on chair' },
  { icon: Star, label: 'Stars & moon' },
  { icon: Mountain, label: 'Mountain' },
  { icon: TreePine, label: 'Tree' },
  { icon: Heart, label: 'Heart' },
]

const IMAGE_MODELS = [
  { value: 'flux-hf', label: 'FLUX.1 Schnell (HuggingFace) - Best Quality' },
  { value: 'pollinations-flux', label: 'Pollinations FLUX - Free Fallback' },
  { value: 'pollinations-turbo', label: 'Pollinations Turbo - Fast' },
]

const ASCII_MODELS = [
  { value: 'resnet18', label: 'ResNet18 - Best' },
  { value: 'vit', label: 'ViT - Experimental' },
  { value: 'legacy', label: 'ResNet18 - Legacy' },
]

const RENDER_MODES = [
  { value: 'auto', label: 'AI Auto-Select (Best Quality)' },
  { value: 'ssim', label: 'Deep Structure (SSIM)' },
  { value: 'portrait', label: 'Portrait (Gradient)' },
  { value: 'cnn', label: 'Standard (CNN)' },
  { value: 'neat', label: 'Neat (Gradient)' },
  { value: 'standard', label: 'Standard (Gradient)' },
  { value: 'high', label: 'High (Gradient)' },
  { value: 'ultra', label: 'Ultra (Gradient)' },
]

function App() {
  const [prompt, setPrompt] = useState('')

  // Generation Settings
  const [width, setWidth] = useState(125)
  const [renderMode, setRenderMode] = useState('auto')
  const [imageModel, setImageModel] = useState('flux-hf')
  const [asciiModel, setAsciiModel] = useState('resnet18')

  // Advanced Options
  const [seed, setSeed] = useState(42)
  const [darkModeInvert, setDarkModeInvert] = useState(false)
  const [autoRouting, setAutoRouting] = useState(true)
  const [semanticPalette, setSemanticPalette] = useState(true)
  const [neuralMapper, setNeuralMapper] = useState(true)
  const [customToken, setCustomToken] = useState('')
  const [useCustomToken, setUseCustomToken] = useState(false)

  // State
  const [loading, setLoading] = useState(false)
  const [ascii, setAscii] = useState('')
  const [image, setImage] = useState('')
  const [logs, setLogs] = useState([])

  const addLog = (msg, type = 'info') => {
    setLogs(prev => [...prev.slice(-8), { msg, type, id: Date.now() }])
  }

  const handleGenerate = async () => {
    if (!prompt.trim()) return
    setLoading(true)
    setLogs([])
    setAscii('')
    setImage('')

    addLog('Starting generation pipeline...')
    addLog(`Prompt: "${prompt}"`)
    addLog(`Mode: ${RENDER_MODES.find(m => m.value === renderMode)?.label}`)

    try {
      const res = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          width,
          quality: renderMode,
          image_model: imageModel,
          ascii_model: asciiModel,
          seed,
          invert: darkModeInvert,
          auto_routing: autoRouting,
          semantic_palette: semanticPalette,
          neural_mapper: neuralMapper,
          custom_token: useCustomToken ? customToken : null
        })
      })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const data = await res.json()
      setAscii(data.ascii || '')
      setImage(data.image_url || '')
      addLog('Generation complete!', 'success')
    } catch (e) {
      addLog(`${e.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <MantineProvider theme={theme} defaultColorScheme="dark">
      <Box style={{ minHeight: '100vh', background: 'var(--mantine-color-dark-8)' }}>
        <Container size="xl" py="xl">
          {/* Header */}
          <Box mb={50} pt="md">
            <Group justify="space-between" align="flex-start" wrap="wrap">
              <Box>
                <Text size="xs" tt="uppercase" fw={700} c="blue.4" mb={8} style={{ letterSpacing: '0.15em' }}>
                  Text-to-Art Pipeline
                </Text>
                <Title
                  order={1}
                  style={{
                    fontSize: 'clamp(2.5rem, 5vw, 3.5rem)',
                    fontWeight: 800,
                    letterSpacing: '-0.03em',
                    lineHeight: 1.1,
                    marginBottom: '0.5rem'
                  }}
                >
                  ASCII Art
                  <Text
                    component="span"
                    inherit
                    style={{
                      background: 'linear-gradient(135deg, #4dabf7 0%, #748ffc 100%)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      marginLeft: '0.3em'
                    }}
                  >
                    Generator
                  </Text>
                </Title>
                <Text c="dimmed" size="lg" maw={500} lw={400}>
                  Turn your ideas into stunning character-based art using the power of FLUX.1 and neural networks.
                </Text>
              </Box>
              <Badge
                variant="light"
                size="md"
                radius="sm"
                color="gray"
                style={{ marginTop: 8, textTransform: 'none', fontWeight: 500 }}
              >
                v2.0 • Production Build
              </Badge>
            </Group>
          </Box>

          <SimpleGrid cols={{ base: 1, lg: 2 }} spacing={40}>
            {/* Left Column - Input & Settings */}
            <Stack gap="xl">
              {/* Creator Section */}
              <Box>
                <Group gap="xs" mb="sm">
                  <Sparkles size={16} className="mantine-primary" />
                  <Text tt="uppercase" size="xs" fw={700} c="dimmed" style={{ letterSpacing: '0.1em' }}>
                    Creator Studio
                  </Text>
                </Group>

                <Paper p="xl" radius="lg" withBorder bg="var(--mantine-color-dark-7)">
                  <Stack gap="md">
                    <Textarea
                      size="lg"
                      label={<Text fw={600} mb={4}>Prompt</Text>}
                      description="Describe what you want to create in detail"
                      placeholder="A cyberpunk city street at night with neon rain..."
                      minRows={3}
                      value={prompt}
                      onChange={(e) => setPrompt(e.currentTarget.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault()
                          handleGenerate()
                        }
                      }}
                      styles={{ input: { fontSize: '1.1rem' } }}
                    />

                    <Group gap="xs">
                      <Group gap={6} align="center">
                        <Lightbulb size={14} color="var(--mantine-color-yellow-5)" />
                        <Text size="xs" fw={600} c="dimmed">TRY AN EXAMPLE</Text>
                      </Group>
                      <Group gap={6}>
                        {EXAMPLES.map(ex => {
                          const IconComponent = ex.icon
                          return (
                            <Button
                              key={ex.label}
                              variant="subtle"
                              color="gray"
                              size="xs"
                              radius="xl"
                              leftSection={<IconComponent size={14} />}
                              onClick={() => setPrompt(ex.label)}
                              styles={{ root: { backgroundColor: 'rgba(255,255,255,0.03)' } }}
                            >
                              {ex.label}
                            </Button>
                          )
                        })}
                      </Group>
                    </Group>
                  </Stack>
                </Paper>
              </Box>

              {/* Configuration Section */}
              <Box>
                <Group gap="xs" mb="sm">
                  <Settings size={16} className="mantine-primary" />
                  <Text tt="uppercase" size="xs" fw={700} c="dimmed" style={{ letterSpacing: '0.1em' }}>
                    Configuration
                  </Text>
                </Group>

                <Paper radius="lg" withBorder overflow="hidden">
                  <Tabs defaultValue="basic" variant="pills" radius="md" p="sm">
                    <Tabs.List style={{ border: 'none' }}>
                      <Tabs.Tab value="basic" leftSection={<Settings size={14} />} fw={600}>
                        Basic
                      </Tabs.Tab>
                      <Tabs.Tab value="models" leftSection={<Cpu size={14} />} fw={600}>
                        Models
                      </Tabs.Tab>
                      <Tabs.Tab value="advanced" leftSection={<Wand2 size={14} />} fw={600}>
                        Advanced
                      </Tabs.Tab>
                    </Tabs.List>

                    <Divider my="sm" color="white" opacity={0.06} />

                    <Tabs.Panel value="basic" p="xs">
                      <Stack gap="lg">
                        <Box>
                          <Group justify="space-between" mb="xs">
                            <Text size="sm" fw={600}>Output Width</Text>
                            <Badge variant="light" color="blue">{width} chars</Badge>
                          </Group>
                          <Slider
                            size="lg"
                            min={30}
                            max={150}
                            step={5}
                            value={width}
                            onChange={setWidth}
                            marks={[
                              { value: 30, label: '30' },
                              { value: 90, label: '90' },
                              { value: 150, label: '150' },
                            ]}
                          />
                        </Box>
                        <Select
                          label={<Text size="sm" fw={600} mb={4}>Render Mode</Text>}
                          description="Algorithm for converting image to ASCII"
                          data={RENDER_MODES}
                          value={renderMode}
                          onChange={setRenderMode}
                          size="md"
                        />
                      </Stack>
                    </Tabs.Panel>

                    <Tabs.Panel value="models" p="xs">
                      <Stack gap="md">
                        <Select
                          label={<Text size="sm" fw={600} mb={4}>Image Generation Model</Text>}
                          data={IMAGE_MODELS}
                          value={imageModel}
                          onChange={setImageModel}
                        />
                        <Select
                          label={<Text size="sm" fw={600} mb={4}>ASCII Mapping Model</Text>}
                          data={ASCII_MODELS}
                          value={asciiModel}
                          onChange={setAsciiModel}
                        />
                        <Paper p="sm" withBorder bg="dark.8">
                          <Switch
                            label={<Text size="sm" fw={500}>Enable Neural Mapper</Text>}
                            description="AI-enhanced character selection"
                            checked={neuralMapper}
                            onChange={(e) => setNeuralMapper(e.currentTarget.checked)}
                          />
                        </Paper>
                        <Box>
                          <Switch
                            label={<Text size="sm" fw={500}>Use Custom HuggingFace Token</Text>}
                            checked={useCustomToken}
                            onChange={(e) => setUseCustomToken(e.currentTarget.checked)}
                            mb="xs"
                          />
                          {useCustomToken && (
                            <Textarea
                              placeholder="hf_..."
                              value={customToken}
                              onChange={(e) => setCustomToken(e.currentTarget.value)}
                            />
                          )}
                        </Box>
                      </Stack>
                    </Tabs.Panel>

                    <Tabs.Panel value="advanced" p="xs">
                      <Stack gap="md">
                        <NumberInput
                          label={<Text size="sm" fw={600} mb={4}>Seed</Text>}
                          value={seed}
                          onChange={setSeed}
                          min={0}
                          max={999999}
                        />
                        <SimpleGrid cols={2}>
                          <Paper p="sm" withBorder bg="dark.8">
                            <Switch
                              label={<Group gap={6}><Moon size={14} /><Text size="sm" fw={500}>Dark Mode</Text></Group>}
                              checked={darkModeInvert}
                              onChange={(e) => setDarkModeInvert(e.currentTarget.checked)}
                            />
                          </Paper>
                          <Paper p="sm" withBorder bg="dark.8">
                            <Switch
                              label={<Group gap={6}><Route size={14} /><Text size="sm" fw={500}>Auto-Routing</Text></Group>}
                              checked={autoRouting}
                              onChange={(e) => setAutoRouting(e.currentTarget.checked)}
                            />
                          </Paper>
                          <Paper p="sm" withBorder bg="dark.8" style={{ gridColumn: 'span 2' }}>
                            <Switch
                              label={<Group gap={6}><Palette size={14} /><Text size="sm" fw={500}>Semantic Palette</Text></Group>}
                              description="Use subject-aware characters"
                              checked={semanticPalette}
                              onChange={(e) => setSemanticPalette(e.currentTarget.checked)}
                            />
                          </Paper>
                        </SimpleGrid>
                      </Stack>
                    </Tabs.Panel>
                  </Tabs>
                </Paper>
              </Box>

              {/* Generate Button */}
              <Button
                size="xl"
                fullWidth
                radius="md"
                gradient={{ from: 'blue', to: 'cyan', deg: 90 }}
                variant="gradient"
                leftSection={loading ? <Loader size={20} color="white" /> : <Sparkles size={20} />}
                onClick={handleGenerate}
                disabled={loading || !prompt.trim()}
                styles={{ root: { transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-2px)' } } }}
              >
                {loading ? 'Generating Art...' : 'Generate ASCII Art'}
              </Button>

              {/* Process Log */}
              {logs.length > 0 && (
                <Paper p="sm" withBorder bg="dark.8">
                  <Group justify="space-between" mb="xs">
                    <Text size="xs" fw={700} c="dimmed" tt="uppercase" style={{ letterSpacing: '0.05em' }}>
                      Thinking Process
                    </Text>
                    <Brain size={14} color="var(--mantine-color-dimmed)" />
                  </Group>
                  <Code block style={{ background: 'transparent', fontSize: 11, lineHeight: 1.6 }}>
                    {logs.map(l => (
                      <div key={l.id} style={{
                        color: l.type === 'error' ? '#ff6b6b' :
                          l.type === 'success' ? '#69db7c' : '#adb5bd',
                        display: 'flex',
                        gap: '8px'
                      }}>
                        <span style={{ opacity: 0.5 }}>›</span> {l.msg}
                      </div>
                    ))}
                  </Code>
                </Paper>
              )}
            </Stack>

            {/* Right Column - Output */}
            <Stack gap="xl">
              <Box>
                <Group gap="xs" mb="sm">
                  <ImageIcon size={16} className="mantine-primary" />
                  <Text tt="uppercase" size="xs" fw={700} c="dimmed" style={{ letterSpacing: '0.1em' }}>
                    Visual Output
                  </Text>
                </Group>

                <Paper p="xs" radius="lg" withBorder bg="var(--mantine-color-dark-8)">
                  <Box
                    style={{
                      aspectRatio: '1',
                      background: 'var(--mantine-color-dark-7)',
                      borderRadius: 'var(--mantine-radius-md)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      overflow: 'hidden',
                      position: 'relative'
                    }}
                  >
                    {image ? (
                      <img src={image} alt="Generated" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
                    ) : (
                      <Stack align="center" gap="xs" c="dimmed">
                        <ImageIcon size={48} strokeWidth={1} />
                        <Text size="sm">Image Visualization</Text>
                      </Stack>
                    )}
                  </Box>
                </Paper>
              </Box>

              <Box style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                <Group justify="space-between" mb="sm">
                  <Group gap="xs">
                    <Terminal size={16} className="mantine-primary" />
                    <Text tt="uppercase" size="xs" fw={700} c="dimmed" style={{ letterSpacing: '0.1em' }}>
                      ASCII Result
                    </Text>
                  </Group>
                  {ascii && (
                    <CopyButton value={ascii}>
                      {({ copied, copy }) => (
                        <Tooltip label={copied ? 'Copied!' : 'Copy raw text'}>
                          <ActionIcon variant="light" onClick={copy} color={copied ? 'teal' : 'gray'} radius="md">
                            {copied ? <Check size={16} /> : <Copy size={16} />}
                          </ActionIcon>
                        </Tooltip>
                      )}
                    </CopyButton>
                  )}
                </Group>

                <Paper flex={1} radius="lg" withBorder overflow="hidden" style={{ display: 'flex', flexDirection: 'column' }}>
                  <Box
                    p="md"
                    style={{
                      background: '#0d1117',
                      flex: 1,
                      minHeight: 400,
                      overflow: 'auto',
                      position: 'relative'
                    }}
                  >
                    {ascii ? (
                      <pre style={{
                        fontFamily: 'JetBrains Mono, Fira Code, monospace',
                        fontSize: 6, // Keep small for detail
                        lineHeight: 1,
                        color: '#58a6ff',
                        margin: 0,
                        whiteSpace: 'pre'
                      }}>
                        {ascii}
                      </pre>
                    ) : (
                      <Center h="100%">
                        <Stack align="center" gap="xs" c="dimmed" style={{ opacity: 0.5 }}>
                          <Terminal size={48} strokeWidth={1} />
                          <Text size="sm">Waiting for input...</Text>
                        </Stack>
                      </Center>
                    )}
                  </Box>
                  {ascii && (
                    <Box p="xs" bg="dark.8" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                      <Group justify="flex-end" gap="xs">
                        <Text size="xs" c="dimmed">{width} chars wide</Text>
                        <Text size="xs" c="dimmed">•</Text>
                        <Text size="xs" c="dimmed">{ascii.split('\n').length} lines</Text>
                      </Group>
                    </Box>
                  )}
                </Paper>
              </Box>
            </Stack>
          </SimpleGrid>
        </Container>
      </Box>
    </MantineProvider>
  )
}

export default App
