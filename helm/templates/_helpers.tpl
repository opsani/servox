{{/*
Identifier for the given installation. Supersedes Release.Name
*/}}
{{- define "servox.identifier" -}}
{{- default .Release.Name .Values.servoUid | required "Either a release name or Values.servoUid must be specified" }}
{{- end }}

{{/*
Name used for resources
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "servox.fullname" -}}
{{- include "servox.identifier" . | printf "%s-%s" .Chart.Name | trunc 63 }}
{{- end }}

{{/*
Namespace to deploy servox into. Must be the same as that of the target workload
*/}}
{{- define "servox.namespace" -}}
{{- default .Release.Namespace .Values.workloadNamespaceName | required "Must specify a namespace for servox installation" }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "servox.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "servox.labels" -}}
helm.sh/chart: {{ include "servox.chart" . }}
{{ include "servox.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "servox.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ include "servox.identifier" . }}
app.kubernetes.io/component: core
{{- end }}
