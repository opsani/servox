{{/*
Copyright 2023 Cisco Systems, Inc. and/or its affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/}}

{{/*
Identifier for the given installation. Supersedes Release.Name
*/}}
{{- define "servox.identifier" -}}
{{- if and (eq .Release.Name "release-name") (empty .Values.servoUid) }}
{{- fail "Either a release name or Values.servoUid must be specified" }}
{{- end }}
{{- default .Release.Name .Values.servoUid }}
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
{{- if and (eq .Release.Namespace "default") (empty .Values.workloadNamespaceName) }}
{{- fail "Must specify the --namespace argument or .Values.workloadNamespaceName (note: namespace 'default' can only be set via .Values.workloadNamespaceName)" }}
{{- end }}
{{- default .Release.Namespace .Values.workloadNamespaceName }}
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
{{- $version := split ":" (include "servox.servoImage" .) }}
app.kubernetes.io/version: {{ print $version._1 | quote }}
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

{{/*
Servo Image
*/}}
{{- define "servox.servoImage" -}}
{{- default (printf "IMAGE_REGISTRY_VALUE/servox:%s" .Chart.AppVersion) .Values.servoImageOverride }}
{{- end }}
